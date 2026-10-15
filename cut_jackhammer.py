"""Cut a recording down to the moments the jackhammer is running.

    python cut_jackhammer.py --input gameplay.mp4 --output cut --example 0:33 --example 0:57

Learns the jackhammer sound from the example times, finds every burst, keeps
each one with --pad seconds either side and writes:

    <name>_jackhammer.mp4   the cut video
    clips.csv               kept ranges in source time; pass it to
                            rust_counter.py --edl to get a matching overlay
    bursts.csv              every detected burst, for checking
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from rustcounter import audio, video
from rustcounter.config import format_timestamp, parse_timestamp
from rustcounter.deps import INSTALL_HELP, setup_tools


def keyframes(ffprobe: str, path: Path) -> list:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    times = []
    for line in out.splitlines():
        pts, _, flags = line.partition(",")
        if "K" in flags and pts not in ("", "N/A"):
            times.append(float(pts))
    return sorted(times)


def offset_expr(keep: list) -> str:
    """setpts expression removing every gap before the current frame."""
    terms, prev_end = [], 0.0
    for a, b in keep:
        gap = a - prev_end
        if gap > 0:
            terms.append(f"{gap:.6f}*gte(T,{a:.6f})")
        prev_end = b
    return "+".join(terms) or "0"


def cut_exact(ffmpeg: str, encoders: set, src: Path, dst: Path, keep: list,
              encoder: str, quality: int) -> None:
    """Frame-exact cut in one pass: drop everything outside `keep`, close the gaps."""
    between = "+".join(f"between(t,{a:.6f},{b:.6f})" for a, b in keep)
    off = offset_expr(keep)
    graph = (f"[0:v]select='{between}',setpts='PTS-({off})/TB'[v];"
             f"[0:a]asetnsamples=n=480,aselect='{between}',asetpts='PTS-({off})/TB',"
             f"aresample=async=1:first_pts=0[a]")
    if encoder == "auto":
        encoder = "hevc_nvenc" if "hevc_nvenc" in encoders else "libx265"
    venc = {
        # Constant QP: NVENC's -cq mode silently capped this footage at ~20 Mbps
        # (35 dB PSNR); QP 16 measured 44 dB, visually transparent.
        "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "constqp",
                       "-qp", str(quality)],
        "libx265": ["-c:v", "libx265", "-crf", str(quality), "-preset", "medium"],
    }[encoder]
    print(f"Encoding with {encoder} (quality {quality}) ...")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y", "-i", str(src),
                    "-filter_complex", graph, "-map", "[v]", "-map", "[a]", *venc,
                    "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", str(dst)], check=True)


def cut_copy(ffmpeg: str, ffprobe: str, src: Path, dst: Path, keep: list) -> list:
    """Lossless cut: pieces start on the keyframe at or before each range."""
    kf = keyframes(ffprobe, src)
    snapped = []
    for a, b in keep:
        a = max([k for k in kf if k <= a + 1e-3] or [0.0])
        if snapped and a <= snapped[-1][1]:
            snapped[-1][1] = max(snapped[-1][1], b)
        else:
            snapped.append([a, b])
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        parts, actual = [], []
        for i, (a, b) in enumerate(snapped):
            part = Path(tmp) / f"part{i:04d}.mp4"
            subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{a + 0.001:.3f}", "-i", str(src),
                            "-t", f"{b - a:.3f}", "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                            "-avoid_negative_ts", "make_zero", str(part)], check=True)
            dur = float(subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                                        "-of", "csv=p=0", str(part)],
                                       capture_output=True, text=True).stdout.strip())
            parts.append(part)
            actual.append((a, a + dur))
            print(f"\r  piece {i + 1}/{len(snapped)}", end="", flush=True)
        print()
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    return actual


def write_ranges(path: Path, ranges: list, header: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for a, b in ranges:
            w.writerow([f"{a:.3f}", f"{b:.3f}"] + ([1] if header[-1] == "speed" else
                                                   [format_timestamp(a), f"{b - a:.2f}"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="cut")
    ap.add_argument("--example", action="append", required=True,
                    help="a time where the jackhammer is running, e.g. 0:33 (give 2 or more)")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds kept before and after each burst")
    ap.add_argument("--sensitivity", type=float, default=14.0,
                    help="dB below the examples' loudness still counted (default 14)")
    ap.add_argument("--similarity", type=float, default=0.95,
                    help="how closely the sound must match the examples, 0-1 (default 0.95)")
    ap.add_argument("--copy", action="store_true",
                    help="no re-encode (zero quality loss); pieces start on keyframes, so they begin earlier")
    ap.add_argument("--encoder", default="auto", choices=["auto", "hevc_nvenc", "libx265"])
    ap.add_argument("--quality", type=int, default=16, help="lower = better and bigger (default 16)")
    ap.add_argument("--dry-run", action="store_true", help="detect and write the CSVs, no video")
    ap.add_argument("--ffmpeg")
    args = ap.parse_args()

    tools = setup_tools(args.ffmpeg, None)
    if not tools.ffmpeg:
        print(INSTALL_HELP["ffmpeg"], file=sys.stderr)
        return 1
    ffprobe = str(Path(tools.ffmpeg).with_name(Path(tools.ffmpeg).name.replace("ffmpeg", "ffprobe")))
    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    duration = video.probe(src).duration

    print("Analysing audio ...")
    feats = audio.features(audio.load_audio(tools.ffmpeg, str(src)))
    tpl = audio.learn(feats, [parse_timestamp(e) for e in args.example])
    bursts = audio.segments(feats, audio.detect(feats, tpl, args.sensitivity, args.similarity))
    keep = audio.pad_and_merge(bursts, args.pad, duration)
    kept = sum(b - a for a, b in keep)
    print(f"{len(bursts)} bursts -> {len(keep)} pieces, keeping {kept:.0f}s of {duration:.0f}s "
          f"({format_timestamp(kept)[3:8]})")
    write_ranges(out / "bursts.csv", bursts, ["start_s", "end_s", "start", "length_s"])

    if args.dry_run:
        write_ranges(out / "clips.csv", keep, ["source_in", "source_out", "speed"])
        print(f"Wrote {out / 'bursts.csv'} and {out / 'clips.csv'} (dry run, no video)")
        return 0

    dst = out / f"{src.stem}_jackhammer.mp4"
    if args.copy:
        keep = cut_copy(tools.ffmpeg, ffprobe, src, dst, keep)
        print(f"Lossless cut keeps {sum(b - a for a, b in keep):.0f}s (pieces start on keyframes)")
    else:
        cut_exact(tools.ffmpeg, tools.encoders, src, dst, keep, args.encoder, args.quality)
    write_ranges(out / "clips.csv", keep, ["source_in", "source_out", "speed"])
    print(f"Wrote {dst}\n      {out / 'clips.csv'}\n      {out / 'bursts.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
