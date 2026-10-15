# rust_counter

Tools for Rust scrap-farming videos.

- **`rust_counter.py`**: reads the `SCRAP +18 (447)` pickups from the HUD with OCR and renders a transparent scrap counter + timer overlay (ProRes 4444 `.mov`) to put on top of the original footage.
- **`cut_jackhammer.py`**: keeps only the moments the jackhammer is running (learned from 2+ example times), with 0.5 s either side.

## Setup

Python 3.10+, [FFmpeg](https://ffmpeg.org), [Tesseract](https://github.com/tesseract-ocr/tesseract).

```
pip install -r requirements.txt
```

## Use

```
# counter + timer overlay (timestamps are in the original recording)
python rust_counter.py --input run.mp4 --output out --starting-scrap 125 --counter-start 0:14 --barrel-start 0:25 --timer-stop 10:00

# cut to the jackhammer bursts
python cut_jackhammer.py --input run.mp4 --output cut --example 0:33 --example 0:57

# overlay matching the cut, reusing the OCR results
python rust_counter.py --input run.mp4 --output cut --events out/scrap_events.csv --edl cut/clips.csv --counter-start 0:14 --barrel-start 0:25
```

- Wrong or missing reading: edit `scrap_events.csv`, re-render with `--events`.
- Run continues in a second recording: `--counter-initial` / `--timer-initial`; `tools/align_clips.py` finds where the two overlap.
- `--help` on either script lists everything else.
