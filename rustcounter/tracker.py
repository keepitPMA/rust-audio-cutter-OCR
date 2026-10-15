"""Turn per-frame OCR readings into validated scrap events.

A new inventory total becomes an event once it has been seen in several
sampled frames. It is then checked arithmetically against the pickup amount
Rust prints next to it:

    SCRAP +17 (142)   fresh notification:  total - previous_total == 17
    SCRAP +35 (160)   same notification, amount accumulated: 35 - 17 == 160 - 142

A total that passes is "confirmed" whatever its size, so large pickups are
never dropped for being unusual. A total that fails is kept as "uncertain"
and flagged for review rather than discarded.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import TrackerConfig, format_timestamp, parse_timestamp

CONFIRMED, UNCERTAIN, REJECTED = "confirmed", "uncertain", "rejected"
# Total not printed by the HUD but derived from the pickup amount; applied.
INFERRED = "inferred"

CSV_FIELDS = ["timestamp_s", "timestamp", "inventory_total", "corrected_scrap",
              "pickup_amount", "ocr_confidence", "status", "use", "frames_seen",
              "note", "raw_text"]


@dataclass
class Event:
    t: float
    total: int | None
    corrected: int
    gain: int | None = None
    confidence: float = 0.0
    status: str = CONFIRMED
    use: bool = True
    frames: int = 0
    note: str = ""
    raw_text: str = ""


@dataclass
class _Candidate:
    total: int
    first_t: float
    last_t: float
    count: int = 0
    strong: int = 0
    best_conf: float = 0.0
    gains: Counter = field(default_factory=Counter)
    matches: Counter = field(default_factory=Counter)
    raw_text: str = ""
    resolved: bool = False


class ScrapTracker:
    def __init__(self, cfg: TrackerConfig, starting_scrap: int, counter_initial: int = 0):
        """counter_initial: scrap already counted in an earlier recording."""
        self.cfg = cfg
        self.starting = starting_scrap
        # The inventory is expected to hold this much when the video begins.
        starting_scrap = starting_scrap + counter_initial
        self.events: list[Event] = []
        self._pending: dict[int, _Candidate] = {}
        # Total and pickup amount of the notification currently trusted.
        self._total = starting_scrap
        self._gain: int | None = None
        # Total printed by the notification on screen; can lag behind _total.
        self._shown_total = starting_scrap
        # Pickup amount seen growing while the printed total stood still.
        self._grown: _Candidate | None = None
        self._current: Event | None = None
        # Same, as of the last arithmetically confirmed event.
        self._anchor = (starting_scrap, None)
        # Last large jump that was flagged but not applied.
        self._held: Event | None = None

    def feed(self, t: float, total: int | None, gain: int | None = None,
             confidence: float = 0.0, match: str = "label", raw_text: str = "") -> None:
        """Feed one sampled frame; total=None means no scrap row was read."""
        self._expire(t)
        if total is None:
            return
        if total in (self._total, self._shown_total):
            if self._gain is None and total == self._total:
                # A notification that was already up when the video began.
                self._gain = gain
            # The same notification is still on screen: no new total. Its
            # pickup amount may still have grown, see _amount_grew.
            if total == self._shown_total and gain is not None and self._gain is not None \
                    and gain > self._gain:
                self._amount_grew(t, gain, confidence, raw_text)
            elif self._current is not None:
                self._current.frames += 1
            return
        cand = self._pending.get(total)
        if cand is None:
            cand = self._pending[total] = _Candidate(total, t, t, raw_text=raw_text)
        cand.last_t = t
        cand.count += 1
        cand.strong += confidence >= self.cfg.min_confidence
        cand.best_conf = max(cand.best_conf, confidence)
        cand.matches[match] += 1
        if gain is not None:
            cand.gains[gain] += 1
        if not cand.resolved and cand.count >= self.cfg.min_frames:
            self._resolve(cand)

    def _amount_grew(self, t: float, gain: int, confidence: float, raw_text: str) -> None:
        """Handle "+35 (1648)" turning into "+53 (1648)".

        Rust sometimes keeps adding to the pickup amount of a notification
        without refreshing the total in parentheses. The real total is then
        the trusted one plus the growth of the amount.
        """
        grown = self._grown
        if grown is None or grown.total != gain:
            # Reuse _Candidate, keyed on the amount instead of the total.
            grown = self._grown = _Candidate(gain, t, t, raw_text=raw_text)
        grown.last_t = t
        grown.count += 1
        grown.strong += confidence >= self.cfg.min_confidence
        grown.best_conf = max(grown.best_conf, confidence)
        if grown.strong < max(3, self.cfg.min_frames):
            return
        total = self._total + gain - self._gain
        flashed = self._pending.pop(total, None)
        if flashed is not None:
            # The HUD did show the new total, just too briefly to count.
            status, t0 = CONFIRMED, flashed.first_t
            note = "total shown only briefly, then the HUD went back to the old one"
        else:
            status, t0 = INFERRED, grown.first_t
            note = (f"HUD total stayed at {self._shown_total}; derived from the pickup "
                    f"amount growing +{self._gain} -> +{gain}")
        event = Event(t=t0, total=total, corrected=total - self.starting, gain=gain,
                      confidence=grown.best_conf, status=status, use=True,
                      frames=grown.count, note=note, raw_text=raw_text)
        self.events.append(event)
        self._total, self._gain, self._current, self._grown = total, gain, event, None
        if status == CONFIRMED:
            self._anchor = (total, gain)

    def finish(self) -> list[Event]:
        self._expire(float("inf"))
        self.events.sort(key=lambda e: e.t)
        return self.events

    def _expire(self, t: float) -> None:
        for total, cand in list(self._pending.items()):
            if t - cand.last_t > self.cfg.window_s:
                del self._pending[total]
                if not cand.resolved:
                    self._emit(cand, REJECTED, False,
                               f"seen in only {cand.count} frame(s)")

    @staticmethod
    def _verified(delta: int, gain: int | None, base_gain: int | None) -> bool:
        if gain is None or delta <= 0:
            return False
        if delta == gain:
            return True
        return base_gain is not None and gain > base_gain and delta == gain - base_gain

    def _resolve(self, cand: _Candidate) -> None:
        cand.resolved = True
        gain = cand.gains.most_common(1)[0][0] if cand.gains else None
        delta = cand.total - self._total
        weak = [] if cand.strong >= self.cfg.min_frames else ["low OCR confidence"]
        if cand.matches["label"] == 0:
            weak.append("row identified by layout, label unreadable")

        current = self._current
        if self._verified(delta, gain, self._gain):
            status = UNCERTAIN if cand.matches["label"] == 0 else CONFIRMED
            if status == CONFIRMED and current is not None and current.status == INFERRED:
                current.note += f"; consistent with the next total {cand.total}"
            self._accept(cand, gain, status, "; ".join(weak))
            return

        # The current value may itself be a misread. If this total follows on
        # from the last confirmed event instead, trust it and demote the other.
        if current is not None and current.status in (UNCERTAIN, INFERRED) and current.use:
            base_total, base_gain = self._anchor
            if self._verified(cand.total - base_total, gain, base_gain):
                current.use = False
                current.status = REJECTED
                current.note = (current.note + "; " if current.note else "") + \
                    f"superseded by verified total {cand.total}"
                self._accept(cand, gain, CONFIRMED, "; ".join(weak))
                return

        # Likewise, a large jump that was held back is vindicated when the next
        # pickup continues exactly from it.
        held = self._held
        if held is not None and self._verified(cand.total - held.total, gain, held.gain):
            held.use = True
            held.note = f"large jump, validated by the following total {cand.total}"
            self._held = None
            self._accept(cand, gain, CONFIRMED, "; ".join(weak))
            return

        if delta <= 0:
            self._emit(cand, REJECTED, False,
                       f"total went down from {self._total}; not applied", gain)
            return
        if gain is None:
            reason = "no pickup amount read"
        elif delta > gain:
            reason = (f"rose by {delta}, {delta - gain} more than the +{gain} shown: scrap gained "
                      "without a notification (e.g. looted from a crate) or a missed pickup")
        else:
            reason = f"+{gain} does not explain the change of {delta} from {self._total}"
        if delta > self.cfg.max_unverified_jump:
            self._held = self._emit(
                cand, UNCERTAIN, False,
                f"LARGE JUMP +{delta} not verified ({reason}); NOT applied, "
                "set use=1 if it is real", gain)
            return
        self._accept(cand, gain, UNCERTAIN, "; ".join([reason] + weak))

    def _accept(self, cand: _Candidate, gain: int | None, status: str, note: str) -> None:
        event = self._emit(cand, status, True, note, gain)
        self._total, self._gain, self._current = cand.total, gain, event
        self._shown_total, self._grown = cand.total, None
        if status == CONFIRMED:
            self._anchor = (cand.total, gain)

    def _emit(self, cand: _Candidate, status: str, use: bool, note: str,
              gain: int | None = None) -> Event:
        if gain is None and cand.gains:
            gain = cand.gains.most_common(1)[0][0]
        event = Event(t=cand.first_t, total=cand.total,
                      corrected=cand.total - self.starting, gain=gain,
                      confidence=cand.best_conf, status=status, use=use,
                      frames=cand.count, note=note, raw_text=cand.raw_text)
        self.events.append(event)
        return event


def write_events(path: str | Path, events: list[Event]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow({
                "timestamp_s": f"{e.t:.3f}",
                "timestamp": format_timestamp(e.t),
                "inventory_total": "" if e.total is None else e.total,
                "corrected_scrap": e.corrected,
                "pickup_amount": "" if e.gain is None else e.gain,
                "ocr_confidence": f"{e.confidence:.0f}",
                "status": e.status,
                "use": int(e.use),
                "frames_seen": e.frames,
                "note": e.note,
                "raw_text": e.raw_text,
            })


def read_events(path: str | Path, starting_scrap: int) -> list[Event]:
    """Load a (possibly hand-edited) events CSV.

    Only timestamp and a value are required per row. "corrected_scrap" is what
    gets displayed; when it is blank it is derived from "inventory_total".
    A blank "use" means the row is used.
    """
    events = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            stamp = row.get("timestamp_s") or row.get("timestamp")
            if not stamp:
                raise ValueError(f"{path} line {n}: missing timestamp")
            total = int(row["inventory_total"]) if row.get("inventory_total") else None
            if row.get("corrected_scrap"):
                corrected = int(row["corrected_scrap"])
            elif total is not None:
                corrected = total - starting_scrap
            else:
                raise ValueError(f"{path} line {n}: needs corrected_scrap or inventory_total")
            use = row.get("use", "")
            events.append(Event(
                t=parse_timestamp(stamp), total=total, corrected=corrected,
                gain=int(row["pickup_amount"]) if row.get("pickup_amount") else None,
                confidence=float(row.get("ocr_confidence") or 0),
                status=row.get("status") or "manual",
                use=use.lower() not in ("0", "false", "no", "n"),
                frames=int(row.get("frames_seen") or 0),
                note=row.get("note", ""), raw_text=row.get("raw_text", ""),
            ))
    events.sort(key=lambda e: e.t)
    return events
