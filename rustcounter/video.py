"""Reading source frames by real timestamp and writing video through FFmpeg.

Game recordings are usually variable frame rate, so a frame's time is taken
from its presentation timestamp, never from frame_index / fps.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .timeline import Clip, schedule

STANDARD_FPS = [24, 25, 30, 48, 50, 60, 90, 120, 144, 240]


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    info = VideoInfo(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, frames / fps)
    cap.release()
    return info


def snap_fps(fps: float) -> float:
    """Snap an average frame rate to the standard rate it was recorded at."""
    nearest = min(STANDARD_FPS, key=lambda r: abs(r - fps))
    return float(nearest) if abs(nearest - fps) / nearest <= 0.03 else round(fps, 3)


class _Reader:
    def __init__(self, path: str | Path, start: float = 0.0):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        if start > 0:
            # Land a little early; callers skip forward to the exact time.
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 0.5) * 1000.0)

    def grab(self) -> float | None:
        """Advance one frame without decoding it to an image; return its time."""
        if not self.cap.grab():
            return None
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def retrieve(self) -> np.ndarray:
        return self.cap.retrieve()[1]

    def close(self) -> None:
        self.cap.release()


def sample_frames(path: str | Path, start: float, end: float, sample_fps: float):
    """Yield (time, frame) roughly sample_fps times per second of video."""
    reader = _Reader(path, start)
    next_t = start
    try:
        while True:
            t = reader.grab()
            if t is None or t >= end:
                break
            if t >= next_t:
                yield t, reader.retrieve()
                next_t = max(next_t + 1.0 / sample_fps, t)
    finally:
        reader.close()


def frames_for_clips(path: str | Path, clips: list[Clip], fps: float):
    """Yield (source_time, frame) for every output frame of a clip list.

    Each output frame shows the latest source frame at or before its time, so
    the result is constant frame rate and stays in sync with the source.
    Callers must not modify a frame: it may be yielded again.
    """
    reader, current = None, None
    for index, target in schedule(clips, fps):
        if index != current:
            if reader is not None:
                reader.close()
            reader, current = _Reader(path, clips[index].source_in), index
            cur = None
            nxt_t = reader.grab()
            nxt = reader.retrieve() if nxt_t is not None else None
        while nxt_t is not None and (cur is None or nxt_t <= target):
            cur = nxt
            nxt_t = reader.grab()
            nxt = reader.retrieve() if nxt_t is not None else None
        yield target, cur
    if reader is not None:
        reader.close()


class FFmpegWriter:
    """Pipe raw frames into an FFmpeg process."""

    def __init__(self, cmd: list[str], log_path: Path):
        self._log = open(log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=self._log, stderr=subprocess.STDOUT)
        self.log_path = log_path

    def write(self, data) -> None:
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError(f"FFmpeg stopped early; see {self.log_path}") from exc

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"FFmpeg failed (exit {code}); see {self.log_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()
            self._log.close()


def alpha_writer(ffmpeg: str, codec: str, out_path: Path, width: int, height: int,
                 fps: float, quality: int, log_path: Path) -> FFmpegWriter:
    """Open a writer for straight-alpha RGBA frames in an alpha-preserving codec."""
    encode = {
        "prores_ks": ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                      "-qscale:v", str(quality), "-vendor", "apl0"],
        "qtrle": ["-c:v", "qtrle", "-pix_fmt", "argb"],
        "png": ["-c:v", "png", "-pix_fmt", "rgba"],
    }[codec]
    cmd = [ffmpeg, "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an",
           *encode, str(out_path)]
    return FFmpegWriter(cmd, log_path)


def preview_writer(ffmpeg: str, out_path: Path, width: int, height: int, fps: float,
                   crf: int, log_path: Path) -> FFmpegWriter:
    cmd = [ffmpeg, "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return FFmpegWriter(cmd, log_path)
