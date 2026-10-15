"""Find where a second recording overlaps the end of a first one.

    python tools/align_clips.py FIRST.mp4 SECOND.mp4

Takes frames from the last seconds of FIRST, finds the same frames in SECOND
and prints the time offset (first_time = second_time + offset). Read-only.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rustcounter import video  # noqa: E402


def signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--tail", type=float, default=6.0, help="seconds at the end of FIRST to match")
    ap.add_argument("--search", type=float, default=0.0, help="seconds of SECOND to search (0 = all)")
    args = ap.parse_args()

    first, second = video.probe(args.first), video.probe(args.second)
    print(f"first : {first.duration:.3f}s   second: {second.duration:.3f}s")

    # Reference frames from the tail of the first recording.
    refs = []
    for t, frame in video.sample_frames(args.first, first.duration - args.tail, first.duration, 4.0):
        refs.append((t, signature(frame)))
    print(f"{len(refs)} reference frames from {refs[0][0]:.3f}s to {refs[-1][0]:.3f}s of the first")

    # Every frame of the second recording.
    end = args.search or second.duration
    times, sigs = [], []
    for t, frame in video.sample_frames(args.second, 0.0, end, 1000.0):
        times.append(t)
        sigs.append(signature(frame))
    stack = np.stack(sigs)
    print(f"{len(times)} frames of the second scanned")

    offsets = []
    for t_ref, sig in refs:
        dist = np.abs(stack - sig).mean(axis=(1, 2))
        best = int(dist.argmin())
        # Runner-up outside the immediate neighbourhood shows how unique the match is.
        far = np.delete(dist, range(max(0, best - 5), min(len(dist), best + 6)))
        print(f"  first {t_ref:9.3f}s  ->  second {times[best]:8.3f}s   diff {dist[best]:5.2f} "
              f"(next best elsewhere {far.min():5.2f})   offset {t_ref - times[best]:.3f}")
        if dist[best] < 0.5 * far.min():
            offsets.append(t_ref - times[best])
    if len(offsets) < 3:
        print("No reliable overlap found.")
        return
    offset = statistics.median(offsets)
    spread = max(offsets) - min(offsets)
    print(f"\noffset = {offset:.3f}s  (first_time = second_time + offset; "
          f"{len(offsets)} matches, spread {spread * 1000:.0f} ms)")
    print(f"the first recording ends at {first.duration - offset:.3f}s of the second")


if __name__ == "__main__":
    main()
