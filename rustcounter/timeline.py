"""What the overlay shows at a given source-video time, and clip mapping.

Everything is keyed on time in the ORIGINAL recording. An edit decision list
maps output frames back to source times, so the overlay can be regenerated
for a cut-down montage without re-running OCR.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path

from .config import parse_timestamp
from .tracker import Event


@dataclass
class Clip:
    source_in: float
    source_out: float
    speed: float = 1.0

    @property
    def output_duration(self) -> float:
        return (self.source_out - self.source_in) / self.speed


class OverlayState:
    def __init__(self, events: list[Event], counter_start: float | None,
                 barrel_start: float | None, timer_stop_s: float | None = None,
                 counter_initial: int = 0, timer_initial: float = 0.0):
        used = sorted((e for e in events if e.use), key=lambda e: e.t)
        self._times = [e.t for e in used]
        self._values = [e.corrected for e in used]
        # Carried over from an earlier recording of the same run.
        self.counter_initial = counter_initial
        self.timer_initial = timer_initial
        # Source time after which the counter holds its value; None = never.
        self.counter_stop: float | None = None
        self.counter_start = counter_start
        self.barrel_start = barrel_start
        self.timer_stop_s = timer_stop_s

    def stop_counter_with_timer(self) -> None:
        """Freeze the counter at the moment the timer reaches its stop value."""
        if self.barrel_start is not None and self.timer_stop_s is not None:
            self.counter_stop = self.barrel_start + self.timer_stop_s - self.timer_initial

    def counter(self, t: float) -> str | None:
        """Counter text at source time t, or None while it is hidden."""
        if self.counter_start is None or t < self.counter_start:
            return None
        if self.counter_stop is not None:
            t = min(t, self.counter_stop)
        i = bisect.bisect_right(self._times, t)
        return str(self._values[i - 1] if i else self.counter_initial)

    def timer(self, t: float) -> str | None:
        """Timer text at source time t, or None before the barrel start."""
        if self.barrel_start is None or t < self.barrel_start:
            return None
        elapsed = t - self.barrel_start + self.timer_initial
        if self.timer_stop_s is not None:
            elapsed = min(elapsed, self.timer_stop_s)
        minutes, seconds = divmod(int(elapsed + 1e-6), 60)
        return f"{minutes:02d}:{seconds:02d}"


def read_edl(path: str | Path) -> list[Clip]:
    """Read a clip list CSV with columns source_in, source_out[, speed].

    Rows are in montage order; timestamps are in source-video time.
    """
    clips = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not row.get("source_in"):
                continue
            clip = Clip(parse_timestamp(row["source_in"]), parse_timestamp(row["source_out"]),
                        float(row.get("speed") or 1.0))
            if clip.source_out <= clip.source_in or clip.speed <= 0:
                raise ValueError(f"{path} line {n}: bad clip {clip}")
            clips.append(clip)
    if not clips:
        raise ValueError(f"{path}: no clips found")
    return clips


def frame_count(clips: list[Clip], fps: float) -> int:
    return int(round(sum(c.output_duration for c in clips) * fps))


def schedule(clips: list[Clip], fps: float):
    """Yield (clip index, source time) for each output frame.

    Output frame k sits at k / fps on the montage timeline, the same clock the
    cut video uses, so many short clips never accumulate rounding drift.
    """
    starts, total = [], 0.0
    for clip in clips:
        starts.append(total)
        total += clip.output_duration
    i = 0
    for k in range(frame_count(clips, fps)):
        t = k / fps
        while i + 1 < len(clips) and t >= starts[i + 1]:
            i += 1
        yield i, clips[i].source_in + (t - starts[i]) * clips[i].speed


def output_times(clips: list[Clip], fps: float):
    """Yield the source time shown by each successive output frame."""
    for _, t in schedule(clips, fps):
        yield t
