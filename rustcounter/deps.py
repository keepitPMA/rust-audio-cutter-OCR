"""Locate FFmpeg and Tesseract and explain how to install them when missing."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

INSTALL_HELP = {
    "ffmpeg": (
        "FFmpeg not found. Install one of:\n"
        "  winget install Gyan.FFmpeg\n"
        "  conda install -c conda-forge ffmpeg\n"
        "or pass --ffmpeg C:\\path\\to\\ffmpeg.exe"
    ),
    "tesseract": (
        "Tesseract not found. Install one of:\n"
        "  winget install UB-Mannheim.TesseractOCR\n"
        "  conda install -c conda-forge tesseract\n"
        "or pass --tesseract C:\\path\\to\\tesseract.exe"
    ),
}


def _env_dirs() -> list[Path]:
    """Binary directories of the Python environment running this script."""
    prefix = Path(sys.prefix)
    return [prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin"]


def _find(name: str, explicit: str | None, env_var: str, extra: list[Path]) -> str | None:
    exe = name + (".exe" if os.name == "nt" else "")
    candidates = [explicit, os.environ.get(env_var)]
    candidates += [str(d / exe) for d in _env_dirs()]
    candidates.append(shutil.which(name))
    candidates += [str(p) for p in extra]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    return None


def find_ffmpeg(explicit: str | None = None) -> str | None:
    return _find("ffmpeg", explicit, "FFMPEG_BINARY", [])


def find_tesseract(explicit: str | None = None) -> str | None:
    extra = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
    ]
    return _find("tesseract", explicit, "TESSERACT_CMD", extra)


def _find_tessdata(tesseract: str) -> str | None:
    exe = Path(tesseract)
    for cand in (
        exe.parent / "tessdata",
        exe.parent.parent / "share" / "tessdata",
        exe.parent.parent.parent / "share" / "tessdata",
    ):
        if (cand / "eng.traineddata").is_file():
            return str(cand)
    return None


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout


@dataclass
class Tools:
    ffmpeg: str | None
    tesseract: str | None
    tessdata: str | None
    encoders: set

    @property
    def alpha_codec(self) -> str | None:
        for codec in ("prores_ks", "qtrle", "png"):
            if codec in self.encoders:
                return codec
        return None


def setup_tools(ffmpeg: str | None = None, tesseract: str | None = None) -> Tools:
    """Find the external tools and make them usable from this process."""
    # A conda env that was not activated keeps its DLLs off PATH.
    extra_path = [str(d) for d in _env_dirs() if d.is_dir()]
    os.environ["PATH"] = os.pathsep.join(extra_path + [os.environ.get("PATH", "")])

    ffmpeg = find_ffmpeg(ffmpeg)
    tesseract = find_tesseract(tesseract)
    tessdata = None
    if tesseract:
        tessdata = os.environ.get("TESSDATA_PREFIX") or _find_tessdata(tesseract)
        if tessdata:
            os.environ["TESSDATA_PREFIX"] = tessdata
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract

    encoders = set()
    if ffmpeg:
        for line in _run([ffmpeg, "-hide_banner", "-encoders"]).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                encoders.add(parts[1])
    return Tools(ffmpeg, tesseract, tessdata, encoders)


def report(tools: Tools) -> str:
    lines = []
    if tools.ffmpeg:
        codec = tools.alpha_codec or "none"
        note = "" if codec == "prores_ks" else "  (ProRes 4444 unavailable)"
        lines.append(f"FFmpeg     : {tools.ffmpeg}")
        lines.append(f"Alpha codec: {codec}{note}")
    else:
        lines.append(INSTALL_HELP["ffmpeg"])
    if tools.tesseract:
        lines.append(f"Tesseract  : {tools.tesseract}")
        lines.append(f"Tessdata   : {tools.tessdata or 'NOT FOUND (set TESSDATA_PREFIX)'}")
    else:
        lines.append(INSTALL_HELP["tesseract"])
    return "\n".join(lines)
