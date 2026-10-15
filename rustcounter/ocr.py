"""Read Rust pickup notifications from the bottom-right HUD.

The notification stack is located by layout first (the lime pickup icon of
each row, falling back to rows of white text), then every row is binarised,
stacked into one image and sent to Tesseract in a single call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract

from .config import OcrConfig

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+()[].,-%x"
_LOOKALIKES = str.maketrans({"5": "S", "0": "O", "1": "I", "8": "B", "4": "A"})


@dataclass
class Geometry:
    """Pixel geometry of the HUD for one frame size."""
    x0: int
    y0: int
    x1: int
    y1: int
    icon: tuple
    text: tuple
    half: int
    scale: float

    @classmethod
    def from_config(cls, cfg: OcrConfig, width: int, height: int) -> "Geometry":
        x0, y0 = int(cfg.region[0] * width), int(cfg.region[1] * height)
        x1, y1 = int(cfg.region[2] * width), int(cfg.region[3] * height)
        icon = (int(cfg.icon_x[0] * width) - x0, int(cfg.icon_x[1] * width) - x0)
        text = (int(cfg.text_x[0] * width) - x0, int(cfg.text_x[1] * width) - x0)
        half = max(4, int(round(cfg.row_half_height * height)))
        return cls(x0, y0, x1, y1, icon, text, half, height / 1440.0)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y1, self.x0:self.x1]


@dataclass
class Row:
    y: int
    text: str = ""
    label: str = ""
    gain: int | None = None
    total: int | None = None
    confidence: float = 0.0
    label_width: int = 0
    match: str = ""  # "label", "layout" or "" when this is not the scrap row


@dataclass
class FrameReading:
    t: float
    rows: list = field(default_factory=list)
    ocr_image: np.ndarray | None = None

    @property
    def scrap(self) -> Row | None:
        hits = [r for r in self.rows if r.match and r.total is not None]
        hits.sort(key=lambda r: (r.match != "label", -r.confidence))
        return hits[0] if hits else None

    @property
    def text(self) -> str:
        return " | ".join(r.text for r in self.rows if r.text)


def _runs(mask_1d: np.ndarray, max_gap: int) -> list:
    """Group the True indices of a 1-D mask into [start, end] runs."""
    runs = []
    for i in np.flatnonzero(mask_1d):
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return runs


def white_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels that are bright and unsaturated, i.e. the white notification text."""
    lo = bgr.min(axis=2).astype(np.int16)
    hi = bgr.max(axis=2).astype(np.int16)
    return (lo > 170) & (hi - lo < 45)


def find_rows(region: np.ndarray, cfg: OcrConfig, geo: Geometry) -> list:
    """Return the y centre (within the region) of every notification row."""
    rows = []
    if cfg.row_detect in ("icon", "auto"):
        hsv = cv2.cvtColor(region[:, geo.icon[0]:geo.icon[1]], cv2.COLOR_BGR2HSV)
        lime = cv2.inRange(hsv, tuple(cfg.icon_hsv_low), tuple(cfg.icon_hsv_high))
        profile = (lime > 0).sum(axis=1)
        for a, b in _runs(profile >= 3, max_gap=int(6 * geo.scale) + 1):
            # The icon is ~32 px tall at 1440p; anything else is scenery.
            if 18 * geo.scale <= b - a + 1 <= 44 * geo.scale:
                rows.append((a + b) // 2)
    if cfg.row_detect == "projection" or (cfg.row_detect == "auto" and not rows):
        mask = white_mask(region[:, geo.text[0]:geo.text[1]])
        profile = mask.sum(axis=1)
        for a, b in _runs(profile >= 12 * geo.scale, max_gap=2):
            if 10 * geo.scale <= b - a + 1 <= 28 * geo.scale:
                rows.append((a + b) // 2)
    return [y for y in rows if geo.half <= y < region.shape[0] - geo.half][:8]


def preprocess_row(strip: np.ndarray, cfg: OcrConfig) -> np.ndarray:
    """Turn one row into black text on a white background for Tesseract."""
    mode = cfg.preprocess
    # The minimum channel is high only for white text: the green row
    # background and the lime icon have a low blue channel.
    gray = strip.min(axis=2) if mode.startswith("minchan") else cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=cfg.upscale, fy=cfg.upscale, interpolation=cv2.INTER_CUBIC)
    if mode.endswith("clahe"):
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 8)).apply(gray)
        return 255 - gray
    if mode.endswith("fixed"):
        _, binary = cv2.threshold(gray, cfg.fixed_threshold, 255, cv2.THRESH_BINARY_INV)
        return binary
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _label_width(strip: np.ndarray, geo: Geometry) -> int:
    """Width in pixels of the first cluster of text ink (the item label)."""
    runs = _runs(white_mask(strip).sum(axis=0) > 0, max_gap=int(1.5 * geo.half))
    return runs[0][1] - runs[0][0] + 1 if runs else 0


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_row_text(text: str) -> tuple:
    """Split "SCRAP +11 (447)" into (label, gain, total)."""
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text.strip())
    head = re.match(r"[^+\d(\[]*", text).group(0)
    label = re.sub(r"[^A-Z ]", "", head.upper()).strip()
    rest = text[len(head):]
    gain = re.search(r"\+\s*(\d+)", rest) or re.match(r"\s*(\d+)", rest)
    # Prefer a closed "(123)"; accept an unclosed "(123" at the end of the row.
    closed = re.findall(r"[(\[{]\s*(\d+)\s*[)\]}]", rest)
    unclosed = re.search(r"[(\[{]\s*(\d+)\s*$", rest)
    total = closed[-1] if closed else unclosed.group(1) if unclosed else None
    return (label,
            int(gain.group(1)) if gain else None,
            int(total) if total is not None else None)


def _classify(row: Row, cfg: OcrConfig, frame_width: int) -> str:
    target = cfg.label.upper()
    label = row.label.replace(" ", "")
    dist = min(_levenshtein(label, target),
               _levenshtein(label.translate(_LOOKALIKES), target))
    if dist <= 1:
        return "label"
    # Layout fallback: a near-miss label whose ink is exactly as wide as the
    # real one, e.g. "SCRAF" or a fading "SCRAPSEP18".
    prefix = _levenshtein(label[:len(target)], target)
    if cfg.label_width > 0 and min(dist, prefix + 1) <= 2:
        expected = cfg.label_width * frame_width
        if abs(row.label_width - expected) <= cfg.label_width_tolerance * expected:
            return "layout"
    return ""


def prepare(frame: np.ndarray, cfg: OcrConfig, geo: Geometry) -> tuple:
    """Find the rows of a frame and build the stacked OCR image.

    Returns (rows, ocr_image, band_height); ocr_image is None when the frame
    shows no pickup text at all.
    """
    region = geo.crop(frame)
    rows, strips = [], []
    for y in find_rows(region, cfg, geo):
        strip = region[y - geo.half:y + geo.half, geo.text[0]:geo.text[1]]
        if white_mask(strip).sum() < 40 * geo.scale ** 2:
            continue  # e.g. the green "BOOSTS ACTIVE" bar
        rows.append(Row(y=y, label_width=_label_width(strip, geo)))
        strips.append(preprocess_row(strip, cfg))
    if not rows:
        return [], None, 0
    pad = 24
    band = strips[0].shape[0] + 2 * pad
    width = strips[0].shape[1] + 2 * pad
    image = np.full((band * len(strips), width), 255, np.uint8)
    for i, strip in enumerate(strips):
        image[i * band + pad:(i + 1) * band - pad, pad:width - pad] = strip
    return rows, image, band


def recognise(rows: list, image: np.ndarray, band: int, cfg: OcrConfig,
              frame_width: int) -> list:
    """Run Tesseract on the stacked image and fill in the rows."""
    data = pytesseract.image_to_data(
        image, lang="eng",
        # The trailing space keeps words apart; without it Tesseract glues
        # "+53 (178)" into one token and reports zero confidence.
        config=f'--oem 1 --psm {cfg.tesseract_psm} -c "tessedit_char_whitelist={WHITELIST} "',
        output_type=pytesseract.Output.DICT,
    )
    words = [[] for _ in rows]
    for text, conf, left, top, height in zip(
            data["text"], data["conf"], data["left"], data["top"], data["height"]):
        if not text.strip() or float(conf) < 0:
            continue
        idx = int((top + height / 2) // band)
        if 0 <= idx < len(rows):
            words[idx].append((left, text.strip(), float(conf)))
    for row, row_words in zip(rows, words):
        row_words.sort()
        row.text = " ".join(w[1] for w in row_words)
        row.label, row.gain, row.total = parse_row_text(row.text)
        # The value matters most: use the weakest word that carries digits.
        numeric = [c for _, t, c in row_words if any(ch.isdigit() for ch in t)]
        confs = numeric or [c for _, _, c in row_words]
        row.confidence = min(confs) if confs else 0.0
        row.match = _classify(row, cfg, frame_width)
    return rows


def read_frame(frame: np.ndarray, t: float, cfg: OcrConfig, geo: Geometry) -> FrameReading:
    rows, image, band = prepare(frame, cfg, geo)
    if image is not None:
        recognise(rows, image, band, cfg, frame.shape[1])
    return FrameReading(t=t, rows=rows, ocr_image=image)
