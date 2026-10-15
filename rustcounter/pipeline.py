"""The three passes: OCR -> events, transparent overlay, debug preview."""
from __future__ import annotations

import bisect
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import ocr, video
from .config import Config, format_timestamp
from .deps import Tools
from .overlay import OverlayRenderer
from .timeline import Clip, OverlayState, frame_count, output_times
from .tracker import CONFIRMED, INFERRED, REJECTED, UNCERTAIN, Event, ScrapTracker


@dataclass
class Sample:
    """What OCR saw in one sampled frame, kept for the debug preview."""
    t: float
    text: str
    total: int | None


def _progress(label: str, done: float, total: float, started: float) -> None:
    pct = 100.0 * done / max(total, 1e-9)
    print(f"\r  {label}: {pct:5.1f}%  ({time.time() - started:.0f}s)", end="", flush=True)


def detect_events(input_path: Path, cfg: Config, info: video.VideoInfo, start: float,
                  end: float, crops_dir: Path | None, counter_initial: int = 0) -> tuple:
    """Run OCR over [start, end) and return (events, samples, stats)."""
    geo = ocr.Geometry.from_config(cfg.ocr, info.width, info.height)
    tracker = ScrapTracker(cfg.tracker, cfg.starting_scrap, counter_initial)
    samples: list[Sample] = []
    first_seen = {}  # total -> (t, region crop, OCR input image)
    stats = {"frames": 0, "with_text": 0, "scrap": 0}

    def recognise(t, rows, image, band, crop):
        if image is not None:
            ocr.recognise(rows, image, band, cfg.ocr, info.width)
        return ocr.FrameReading(t, rows, image), crop

    def consume(future) -> None:
        reading, crop = future.result()
        stats["frames"] += 1
        stats["with_text"] += bool(reading.rows)
        row = reading.scrap
        if row is None:
            tracker.feed(reading.t, None)
        else:
            stats["scrap"] += 1
            tracker.feed(reading.t, row.total, row.gain, row.confidence, row.match, row.text)
            if row.total not in first_seen:
                first_seen[row.total] = (reading.t, crop, reading.ocr_image)
        samples.append(Sample(reading.t, reading.text, row.total if row else None))

    started = time.time()
    pending = deque()
    with ThreadPoolExecutor(max(1, cfg.ocr.workers)) as pool:
        for t, frame in video.sample_frames(input_path, start, end, cfg.ocr.sample_fps):
            rows, image, band = ocr.prepare(frame, cfg.ocr, geo)
            crop = geo.crop(frame).copy() if image is not None else None
            pending.append(pool.submit(recognise, t, rows, image, band, crop))
            while len(pending) > 4 * cfg.ocr.workers:
                consume(pending.popleft())
            _progress("OCR", t - start, end - start, started)
        while pending:
            consume(pending.popleft())
    print()
    events = tracker.finish()

    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for old in crops_dir.glob("event_*.png"):
            old.unlink()
        for i, event in enumerate(events, 1):
            seen = first_seen.get(event.total)
            if seen is None:
                continue
            _, crop, ocr_image = seen
            name = f"event_{i:03d}_{event.t:08.3f}s_total{event.total}_{event.status}"
            cv2.imwrite(str(crops_dir / f"{name}_hud.png"), crop)
            cv2.imwrite(str(crops_dir / f"{name}_ocr_input.png"), ocr_image)
    return events, samples, stats


def render_overlay(tools: Tools, cfg: Config, state: OverlayState, clips: list[Clip],
                   width: int, height: int, fps: float, out_dir: Path,
                   png_frames: bool) -> Path:
    """Write the transparent counter/timer video (and optionally PNG frames)."""
    codec = tools.alpha_codec
    out_path = out_dir / "counter_overlay.mov"
    renderer = OverlayRenderer(cfg.overlay, width, height)
    frames_dir = out_dir / "overlay_frames"
    if png_frames:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)

    total = frame_count(clips, fps)
    started = time.time()
    last_frame, last_bytes, last_png = None, None, None
    with video.alpha_writer(tools.ffmpeg, codec, out_path, width, height, fps,
                            cfg.overlay.prores_quality, out_dir / "ffmpeg_overlay.log") as writer:
        for n, t in enumerate(output_times(clips, fps)):
            frame = renderer.frame(state.counter(t), state.timer(t))
            changed = frame is not last_frame
            if changed:
                last_frame, last_bytes = frame, frame.tobytes()
            writer.write(last_bytes)
            if png_frames:
                png = frames_dir / f"overlay_{n:06d}.png"
                if changed or last_png is None:
                    cv2.imwrite(str(png), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA),
                                [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    last_png = png
                else:
                    shutil.copyfile(last_png, png)
            if n % 30 == 0:
                _progress("Overlay", n, total, started)
    _progress("Overlay", total, total, started)
    print()
    return out_path


def _draw_debug(canvas: np.ndarray, frame: np.ndarray, geo: ocr.Geometry, scale: float,
                t: float, sample: Sample | None, state: OverlayState) -> None:
    """Outline the OCR region and show what was read from it."""
    p0 = (int(geo.x0 * scale), int(geo.y0 * scale))
    p1 = (int(geo.x1 * scale) - 1, int(geo.y1 * scale) - 1)
    cv2.rectangle(canvas, p0, p1, (0, 255, 255), 1)

    # Enlarged copy of the OCR region so the small HUD text stays readable.
    inset_h = int(canvas.shape[0] * 0.42)
    crop = geo.crop(frame)
    inset_w = int(crop.shape[1] * inset_h / crop.shape[0])
    inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    x, y = 12, int(canvas.shape[0] * 0.30)
    canvas[y:y + inset_h, x:x + inset_w] = inset
    cv2.rectangle(canvas, (x, y), (x + inset_w, y + inset_h), (0, 255, 255), 1)

    lines = [f"src {format_timestamp(t)}",
             f"OCR: {sample.text if sample and sample.text else '-'}",
             f"scrap total read: {sample.total if sample and sample.total is not None else '-'}",
             f"shown: counter {state.counter(t) or 'hidden'}  timer {state.timer(t) or 'hidden'}"]
    size = canvas.shape[0] / 1300.0
    # Dark panel behind the text so it stays readable on snow.
    top, bottom = y + inset_h + 2, y + inset_h + int((len(lines) + 0.5) * 30 * size) + 4
    right = min(canvas.shape[1], x + int(canvas.shape[1] * 0.5))
    canvas[top:bottom, x:right] = canvas[top:bottom, x:right] // 4
    for i, line in enumerate(lines):
        org = (x, y + inset_h + int((i + 1) * 30 * size) + 4)
        cv2.putText(canvas, line, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 255), 1, cv2.LINE_AA)


def render_preview(tools: Tools, cfg: Config, input_path: Path, info: video.VideoInfo,
                   state: OverlayState, clips: list[Clip], samples: list[Sample],
                   fps: float, out_dir: Path) -> Path:
    """Write the gameplay with the overlay (and OCR debug view) burned in."""
    height = min(cfg.preview.height, info.height) // 2 * 2
    width = int(round(info.width * height / info.height)) // 2 * 2
    scale = height / info.height
    renderer = OverlayRenderer(cfg.overlay, width, height)
    geo = ocr.Geometry.from_config(cfg.ocr, info.width, info.height)
    sample_times = [s.t for s in samples]
    out_path = out_dir / "debug_preview.mp4"

    total = frame_count(clips, fps)
    started = time.time()
    with video.preview_writer(tools.ffmpeg, out_path, width, height, fps, cfg.preview.crf,
                              out_dir / "ffmpeg_preview.log") as writer:
        for n, (t, frame) in enumerate(video.frames_for_clips(input_path, clips, fps)):
            canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if cfg.preview.show_debug:
                i = bisect.bisect_right(sample_times, t)
                sample = samples[i - 1] if i and t - sample_times[i - 1] < 0.5 else None
                _draw_debug(canvas, frame, geo, scale, t, sample, state)
            renderer.blend_onto(canvas, state.counter(t), state.timer(t))
            writer.write(canvas.tobytes())
            if n % 30 == 0:
                _progress("Preview", n, total, started)
    _progress("Preview", total, total, started)
    print()
    return out_path


def summarise(events: list[Event], stats: dict | None, starting_scrap: int) -> str:
    lines = []
    if stats:
        lines.append(f"Sampled {stats['frames']} frames: {stats['with_text']} showed pickup "
                     f"text, {stats['scrap']} had a readable scrap row.")
    counts = {s: sum(e.status == s for e in events)
              for s in (CONFIRMED, INFERRED, UNCERTAIN, REJECTED)}
    lines.append(f"Events: {counts[CONFIRMED]} confirmed, {counts[INFERRED]} inferred, "
                 f"{counts[UNCERTAIN]} uncertain, {counts[REJECTED]} rejected.")
    if events:
        lines.append(f"{'time':>12}  {'inventory':>9}  {'shown':>5}  {'pickup':>6}  "
                     f"{'conf':>4}  {'frames':>6}  {'status':<9}  note")
        for e in events:
            gain = "" if e.gain is None else f"+{e.gain}"
            total = "" if e.total is None else e.total
            mark = "" if e.use else "  [not applied]"
            lines.append(f"{format_timestamp(e.t):>12}  {total:>9}  {e.corrected:>5}  "
                         f"{gain:>6}  {e.confidence:>4.0f}  {e.frames:>6}  {e.status:<9}  "
                         f"{e.note}{mark}")
    used = [e for e in events if e.use]
    if used:
        lines.append(f"Final counter value: {used[-1].corrected} "
                     f"(inventory {used[-1].total}, minus {starting_scrap} starting scrap).")
    review = [e for e in events if e.status != CONFIRMED]
    if review:
        lines.append(f"REVIEW: {len(review)} event(s) are not confirmed. Check them in "
                     "scrap_events.csv, edit if needed, then re-render with --events.")
    return "\n".join(lines)
