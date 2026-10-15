"""Render the counter and timer as transparent RGBA images with Pillow."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import OverlayConfig, TextStyle

PROJECT_FONTS = Path(__file__).resolve().parent.parent / "fonts"


def _font_dirs() -> list[Path]:
    dirs = [PROJECT_FONTS]
    if os.name == "nt":
        dirs.append(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")
        dirs.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts")
    dirs += [Path.home() / ".fonts", Path.home() / "Library/Fonts",
             Path("/Library/Fonts"), Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=1)
def _installed_fonts() -> dict:
    """Map 'family style' (lower case, no punctuation) to a font file."""
    found = {}
    for directory in _font_dirs():
        for path in directory.rglob("*"):
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            try:
                family, style = ImageFont.truetype(str(path), 20).getname()
            except Exception:
                continue
            key = re.sub(r"[^a-z0-9]", "", f"{family}{style}".lower())
            found.setdefault(key, path)
    return found


def find_font(cfg: OverlayConfig) -> tuple:
    """Return (path, variation) for the first preferred font that exists."""
    if cfg.font_path:
        return Path(cfg.font_path), None
    fonts = _installed_fonts()
    for name in cfg.font_preference:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in fonts:
            return fonts[key], None
        # Variable fonts ship one file for every weight (e.g. Inter from
        # Google Fonts): pick the weight by name.
        family, _, style = name.rpartition(" ")
        for font_key, path in fonts.items():
            if not family or not font_key.startswith(re.sub(r"[^a-z0-9]", "", family.lower())):
                continue
            try:
                names = ImageFont.truetype(str(path), 20).get_variation_names()
            except Exception:
                continue
            for variation in names:
                if variation.decode().lower().replace(" ", "") == style.lower():
                    return path, variation
    return None, None


def load_font(cfg: OverlayConfig, size: int) -> ImageFont.FreeTypeFont:
    path, variation = find_font(cfg)
    if path is None:
        return ImageFont.load_default(size)
    font = ImageFont.truetype(str(path), size)
    if variation:
        font.set_variation_by_name(variation)
    return font


class OverlayRenderer:
    """Draws both texts for one output resolution.

    Rendering is cached per text, since the counter changes rarely and the
    timer once a second.
    """

    def __init__(self, cfg: OverlayConfig, width: int, height: int):
        self.cfg, self.width, self.height = cfg, width, height
        self._fonts = {}
        self._patch = lru_cache(maxsize=256)(self._render_patch)
        self._last_key, self._last_frame = None, None

    def font_name(self) -> str:
        path, variation = find_font(self.cfg)
        if path is None:
            return "Pillow default (no preferred font found)"
        return f"{path.name}" + (f" [{variation.decode()}]" if variation else "")

    def _font(self, style: TextStyle) -> ImageFont.FreeTypeFont:
        size = max(6, int(round(style.font_size * self.height)))
        if size not in self._fonts:
            self._fonts[size] = load_font(self.cfg, size)
        return self._fonts[size]

    def _layout(self, text: str, font) -> tuple:
        """Return per-character x centres and the total width."""
        digit_w = max(font.getlength(d) for d in "0123456789")
        gap = self.cfg.letter_spacing * font.size
        centres, x = [], 0.0
        for ch in text:
            w = digit_w if (ch.isdigit() and self.cfg.tabular_digits) else font.getlength(ch)
            centres.append(x + w / 2)
            x += w + gap
        return centres, max(x - gap, 0.0)

    def _render_patch(self, text: str, which: str) -> tuple:
        """Render one text; returns (x, y, rgba) placed in frame coordinates."""
        style = getattr(self.cfg, which)
        font = self._font(style)
        centres, text_w = self._layout(text, font)
        ascent, descent = font.getmetrics()
        cfg = self.cfg
        blur = cfg.shadow_blur * self.height
        spread = int(round(cfg.shadow_spread * self.height))
        off_x = cfg.shadow_offset[0] * self.height
        off_y = cfg.shadow_offset[1] * self.height
        pad = int(np.ceil(3 * blur + spread + max(abs(off_x), abs(off_y)))) + 2
        size = (int(np.ceil(text_w)) + 2 * pad, ascent + descent + 2 * pad)

        def draw_text(fill, dx=0.0, dy=0.0) -> Image.Image:
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for ch, cx in zip(text, centres):
                draw.text((pad + cx + dx, pad + dy), ch, font=font, fill=fill, anchor="ma")
            return layer

        def shadow(radius: float, opacity: float, grow: int) -> Image.Image:
            alpha = draw_text((0, 0, 0, 255), off_x, off_y).getchannel("A")
            if grow > 0:
                alpha = alpha.filter(ImageFilter.MaxFilter(2 * grow + 1))
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius)).point(lambda a: int(a * opacity))
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            layer.putalpha(alpha)
            return layer

        patch = shadow(blur, cfg.shadow_opacity, spread)
        if cfg.contact_shadow_opacity > 0:
            patch = Image.alpha_composite(patch, shadow(
                cfg.contact_shadow_blur * self.height, cfg.contact_shadow_opacity, 0))
        patch = Image.alpha_composite(patch, draw_text(tuple(style.color)))

        anchor_x = style.x * self.width
        left = {"left": anchor_x, "right": anchor_x - text_w,
                "center": anchor_x - text_w / 2}[style.align]
        return int(round(left)) - pad, int(round(style.y * self.height)) - pad, np.asarray(patch)

    def patches(self, counter: str | None, timer: str | None) -> list:
        out = []
        if counter is not None:
            out.append(self._patch(counter, "counter"))
        if timer is not None:
            out.append(self._patch(timer, "timer"))
        return out

    def frame(self, counter: str | None, timer: str | None) -> np.ndarray:
        """Full transparent RGBA frame (straight alpha)."""
        key = (counter, timer)
        if key != self._last_key:
            canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            for x, y, rgba in self.patches(counter, timer):
                canvas.alpha_composite(Image.fromarray(rgba), (max(x, 0), max(y, 0)))
            self._last_key, self._last_frame = key, np.asarray(canvas)
        return self._last_frame

    def blend_onto(self, bgr: np.ndarray, counter: str | None, timer: str | None) -> None:
        """Alpha-blend the overlay onto a BGR frame of this renderer's size, in place."""
        for x, y, rgba in self.patches(counter, timer):
            h, w = rgba.shape[:2]
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, bgr.shape[1]), min(y + h, bgr.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
            alpha = src[..., 3:4] / 255.0
            dst = bgr[y0:y1, x0:x1].astype(np.float32)
            bgr[y0:y1, x0:x1] = (src[..., 2::-1] * alpha + dst * (1 - alpha)).astype(np.uint8)
