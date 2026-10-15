"""Generate a scrap counter and timer overlay for a Rust gameplay recording.

    python rust_counter.py --input gameplay.mp4 --output output \
        --starting-scrap 125 --barrel-start 00:12.500 --counter-start 00:08.000 \
        --duration 30

All timestamps are in the time of the ORIGINAL recording.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rustcounter import pipeline, video
from rustcounter.config import dump_config, load_config, parse_timestamp
from rustcounter.deps import INSTALL_HELP, report, setup_tools
from rustcounter.overlay import OverlayRenderer
from rustcounter.timeline import Clip, OverlayState, read_edl
from rustcounter.tracker import read_events, write_events


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="source video")
    ap.add_argument("--output", default="output", help="output directory")
    ap.add_argument("--starting-scrap", type=int, help="scrap in the inventory when recording began (default 125)")
    ap.add_argument("--barrel-start", help="source time the timer starts at 00:00; omit to hide the timer")
    ap.add_argument("--counter-start", default="0", help="source time the counter first appears (default 0)")
    ap.add_argument("--duration", type=float, help="seconds of source video to process (default: all)")
    ap.add_argument("--start", default="0", help="source time to start processing from (default 0)")
    ap.add_argument("--counter-initial", type=int, default=0,
                    help="continue a run: counter value carried over from the previous recording")
    ap.add_argument("--timer-initial", default="0",
                    help="continue a run: time already on the timer at --barrel-start, e.g. 11:49.567")
    ap.add_argument("--timer-stop", help="freeze the timer at this elapsed time, e.g. 10:00 (default: never)")
    ap.add_argument("--stop-counter-with-timer", action="store_true",
                    help="with --timer-stop: also freeze the counter at that moment")
    ap.add_argument("--config", help="JSON file overriding the default settings")
    ap.add_argument("--events", help="render from this (edited) events CSV instead of running OCR")
    ap.add_argument("--edl", help="clip list CSV (source_in, source_out[, speed]) to render a cut-down overlay")
    ap.add_argument("--png-frames", action="store_true", help="also export overlay_frames/*.png")
    ap.add_argument("--no-overlay", action="store_true", help="skip counter_overlay.mov")
    ap.add_argument("--no-preview", action="store_true", help="skip debug_preview.mp4")
    ap.add_argument("--no-debug-viz", action="store_true", help="preview without the OCR debug view")
    ap.add_argument("--fps", type=float, help="overlay frame rate (default: source rate)")
    ap.add_argument("--resolution", help="overlay resolution WxH (default: source resolution)")
    ap.add_argument("--ffmpeg", help="path to ffmpeg")
    ap.add_argument("--tesseract", help="path to tesseract")
    ap.add_argument("--check", action="store_true", help="report on FFmpeg/Tesseract and exit")
    ap.add_argument("--dump-config", metavar="PATH", help="write the effective settings to a JSON file and exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.starting_scrap is not None:
        cfg.starting_scrap = args.starting_scrap
    if args.timer_stop:
        cfg.overlay.timer_stop_s = parse_timestamp(args.timer_stop)
    if args.fps:
        cfg.overlay.fps = args.fps
    if args.resolution:
        cfg.overlay.width, cfg.overlay.height = (int(v) for v in args.resolution.lower().split("x"))
    if args.no_debug_viz:
        cfg.preview.show_debug = False

    if args.dump_config:
        dump_config(cfg, args.dump_config)
        print(f"Wrote {args.dump_config}")
        return 0

    tools = setup_tools(args.ffmpeg, args.tesseract)
    print(report(tools))
    if args.check:
        return 0 if tools.ffmpeg and tools.tesseract and tools.tessdata else 1
    if not args.input:
        print("error: --input is required", file=sys.stderr)
        return 2

    need_video = not (args.no_overlay and args.no_preview)
    if not args.events and not (tools.tesseract and tools.tessdata):
        print("\n" + INSTALL_HELP["tesseract"], file=sys.stderr)
        return 1
    if need_video and not tools.ffmpeg:
        print("\n" + INSTALL_HELP["ffmpeg"], file=sys.stderr)
        return 1
    if not args.no_overlay and tools.alpha_codec is None:
        print("error: this FFmpeg has no alpha-capable encoder (prores_ks, qtrle or png). "
              "Install a full FFmpeg build, or pass --no-overlay.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video.probe(input_path)
    start = parse_timestamp(args.start)
    end = min(info.duration, start + args.duration) if args.duration else info.duration
    fps = cfg.overlay.fps or video.snap_fps(info.fps)
    width = cfg.overlay.width or info.width
    height = cfg.overlay.height or info.height
    print(f"\nInput      : {input_path.name}  {info.width}x{info.height}, "
          f"avg {info.fps:.2f} fps, {info.duration:.1f}s")
    print(f"Processing : {start:.3f}s -> {end:.3f}s of the source")
    print(f"Overlay    : {width}x{height} @ {fps:g} fps, font {OverlayRenderer(cfg.overlay, width, height).font_name()}")

    events_path = out_dir / "scrap_events.csv"
    samples, stats = [], None
    if args.events:
        events = read_events(args.events, cfg.starting_scrap)
        print(f"\nLoaded {len(events)} events from {args.events} (OCR skipped)")
    else:
        print()
        events, samples, stats = pipeline.detect_events(
            input_path, cfg, info, start, end, out_dir / "debug_crops", args.counter_initial)
        write_events(events_path, events)
    print(pipeline.summarise(events, stats, cfg.starting_scrap))

    state = OverlayState(events, parse_timestamp(args.counter_start),
                         parse_timestamp(args.barrel_start) if args.barrel_start else None,
                         cfg.overlay.timer_stop_s, args.counter_initial,
                         parse_timestamp(args.timer_initial))
    if args.stop_counter_with_timer:
        state.stop_counter_with_timer()
    clips = read_edl(args.edl) if args.edl else [Clip(start, end)]

    print()
    written = [] if args.events else [events_path, out_dir / "debug_crops"]
    if not args.no_overlay:
        written.append(pipeline.render_overlay(tools, cfg, state, clips, width, height, fps,
                                               out_dir, args.png_frames))
        if args.png_frames:
            written.append(out_dir / "overlay_frames")
        if tools.alpha_codec != "prores_ks":
            print(f"  note: ProRes 4444 unavailable, used '{tools.alpha_codec}' (also keeps alpha)")
    if not args.no_preview:
        written.append(pipeline.render_preview(tools, cfg, input_path, info, state, clips,
                                               samples, fps, out_dir))
    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
