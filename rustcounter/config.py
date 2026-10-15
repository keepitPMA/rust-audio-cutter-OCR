"""Configuration dataclasses, JSON overrides and timestamp helpers.

Every position/size is a fraction of the frame, so the same config works at
any resolution with the same aspect ratio and HUD scale.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class OcrConfig:
    # Bottom-right notification stack, as fractions of the frame: x0, y0, x1, y1.
    region: list = field(default_factory=lambda: [0.8203, 0.5200, 1.0000, 0.8750])
    # Column holding the lime "download" icon of each pickup row (x0, x1).
    icon_x: list = field(default_factory=lambda: [0.8379, 0.8563])
    # Columns holding the row text, from the item label to the "(total)" (x0, x1).
    text_x: list = field(default_factory=lambda: [0.8570, 0.9890])
    # Half the height of one notification row, as a fraction of frame height.
    row_half_height: float = 0.0125
    # HSV range (OpenCV: H 0-180) of the lime pickup icon.
    icon_hsv_low: list = field(default_factory=lambda: [30, 150, 170])
    icon_hsv_high: list = field(default_factory=lambda: [50, 255, 255])
    # "icon" = locate rows by their icon, "projection" = by white text rows,
    # "auto" = icon first, projection when no icon is found.
    row_detect: str = "auto"
    # Preprocessing: "minchan_otsu", "gray_otsu", "minchan_fixed", "minchan_clahe".
    preprocess: str = "minchan_fixed"
    upscale: float = 3.0
    fixed_threshold: int = 150
    sample_fps: float = 8.0
    label: str = "SCRAP"
    # Width of the label's ink as a fraction of frame width; used to recognise
    # the row by layout when the label itself is misread. 0 disables the check.
    label_width: float = 0.0277
    label_width_tolerance: float = 0.12
    tesseract_psm: int = 6
    workers: int = 4


@dataclass
class TrackerConfig:
    # Sampled frames a new total must be seen in before it becomes an event.
    min_frames: int = 2
    # Seconds a candidate may go unseen before it is dropped.
    window_s: float = 2.0
    # OCR confidence (0-100) below which a sighting counts as weak.
    min_confidence: float = 50.0
    # Jumps above this that cannot be verified arithmetically are flagged and
    # NOT applied automatically. Verified jumps of any size are accepted.
    max_unverified_jump: int = 150


@dataclass
class TextStyle:
    # Anchor point as fractions of the frame; "align" says which edge of the
    # text sits on it.
    x: float = 0.03
    y: float = 0.045
    align: str = "left"
    # Font size as a fraction of frame height.
    font_size: float = 0.058
    color: list = field(default_factory=lambda: [255, 255, 255, 255])


@dataclass
class OverlayConfig:
    # Output resolution; 0 = same as the source video.
    width: int = 0
    height: int = 0
    # Output frame rate; 0 = source rate snapped to the nearest standard rate.
    fps: float = 0.0
    font_path: str = ""
    font_preference: list = field(default_factory=lambda: [
        "Inter SemiBold", "Montserrat SemiBold", "Bahnschrift SemiBold", "Segoe UI Semibold",
        "Arial Bold", "DejaVu Sans Bold",
    ])
    # x=0.915 leaves ~8.5% of the width free on the right for the pictogram.
    counter: TextStyle = field(default_factory=lambda: TextStyle(x=0.915, align="right"))
    timer: TextStyle = field(default_factory=lambda: TextStyle(x=0.03, align="left"))
    # Extra space between characters, as a fraction of the font size.
    letter_spacing: float = 0.015
    # Wide diffuse shadow. Blur, spread and offset are fractions of frame
    # height; spread thickens the shape before blurring so the halo stays
    # visible on bright backgrounds.
    shadow_opacity: float = 0.7
    shadow_blur: float = 0.014
    shadow_spread: float = 0.002
    shadow_offset: list = field(default_factory=lambda: [0.0, 0.003])
    # Tight shadow under the wide one; keeps the glyph edges crisp. 0 = off.
    contact_shadow_opacity: float = 0.45
    contact_shadow_blur: float = 0.003
    # Fixed-width digits stop the text jittering as values change.
    tabular_digits: bool = True
    # Freeze the timer at this many seconds (e.g. 600); null = never stop.
    timer_stop_s: float | None = None
    prores_quality: int = 11


@dataclass
class PreviewConfig:
    height: int = 720
    crf: int = 23
    show_debug: bool = True


@dataclass
class Config:
    starting_scrap: int = 125
    ocr: OcrConfig = field(default_factory=OcrConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)


def _merge(obj, data: dict, path: str = "") -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def parse_timestamp(text: str | float | int) -> float:
    """Parse "SS.mmm", "MM:SS.mmm" or "HH:MM:SS.mmm" into seconds."""
    if isinstance(text, (int, float)):
        return float(text)
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Bad timestamp: {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
