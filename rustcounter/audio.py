"""Find jackhammer bursts in a recording's audio.

A jackhammer hit is about a second of loud, continuous rattle at ~25 hits per
second. Each analysis window is described by its loudness, its spectral
balance and how strongly its loudness repeats at 15-40 Hz; a window counts as
jackhammer when all three are close to the examples you point at.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

SR = 16000
HOP = 80            # 5 ms loudness envelope
WIN = 0.30          # analysis window, seconds
STEP = 0.05         # analysis step, seconds
BANDS = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def load_audio(ffmpeg: str, path: str) -> np.ndarray:
    """Decode the first audio track to mono 16 kHz floats."""
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


@dataclass
class Features:
    t: np.ndarray        # window centre, seconds
    level: np.ndarray    # dB
    shape: np.ndarray    # band energy shares, rows sum to 1
    rhythm: np.ndarray   # autocorrelation peak at 15-40 Hz, 0-1


def features(x: np.ndarray) -> Features:
    n_env = (len(x) - 2 * HOP) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, 2 * HOP)[::HOP][:n_env]
    env = np.sqrt((frames ** 2).mean(axis=1))

    nfft = 512
    stft = np.lib.stride_tricks.sliding_window_view(x, nfft)[::160]
    spec = np.abs(np.fft.rfft(stft * np.hanning(nfft), axis=1))
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    band = np.stack([spec[:, (freqs >= a) & (freqs < b)].sum(axis=1) for a, b in BANDS], axis=1)

    per_win = int(WIN * SR / HOP)
    per_step = int(STEP * SR / HOP)
    lags = np.arange(int(0.025 * SR / HOP), int(0.066 * SR / HOP) + 1)
    ts, levels, shapes, rhythms = [], [], [], []
    for start in range(0, len(env) - per_win, per_step):
        e = env[start:start + per_win]
        t0 = start * HOP / SR
        levels.append(20 * np.log10(np.sqrt((e ** 2).mean()) + 1e-9))
        d = e - e.mean()
        denom = (d * d).sum() + 1e-12
        rhythms.append(max((d[:-lag] * d[lag:]).sum() / denom for lag in lags))
        s = band[int(t0 * 100):int((t0 + WIN) * 100)].sum(axis=0)
        shapes.append(s / (s.sum() + 1e-12))
        ts.append(t0 + WIN / 2)
    return Features(np.array(ts), np.array(levels), np.array(shapes), np.array(rhythms))


@dataclass
class Template:
    level: float
    shape: np.ndarray
    rhythm: float


def learn(f: Features, examples: list, search: float = 1.5) -> Template:
    """Average the loudest windows around each example time."""
    picked = []
    for t in examples:
        near = np.flatnonzero(np.abs(f.t - t) <= search)
        if len(near) == 0:
            raise ValueError(f"example {t:.2f}s is outside the audio")
        loud = near[f.level[near] >= f.level[near].max() - 6]
        picked.extend(loud)
    picked = np.array(picked)
    shape = f.shape[picked].mean(axis=0)
    return Template(float(np.median(f.level[picked])), shape / shape.sum(),
                    float(np.median(f.rhythm[picked])))


def detect(f: Features, tpl: Template, level_margin: float = 14.0,
           min_similarity: float = 0.95, min_rhythm: float = 0.25) -> np.ndarray:
    """Boolean per analysis window: does it sound like the template?"""
    cos = (f.shape @ tpl.shape) / (np.linalg.norm(f.shape, axis=1) * np.linalg.norm(tpl.shape))
    return (f.level >= tpl.level - level_margin) & (cos >= min_similarity) & \
        (f.rhythm >= min(min_rhythm, 0.6 * tpl.rhythm))


def segments(f: Features, active: np.ndarray, fill_gap: float = 0.35,
             min_len: float = 0.25) -> list:
    """Turn active windows into (start, end) bursts in seconds."""
    runs = []
    for i in np.flatnonzero(active):
        a, b = f.t[i] - STEP / 2, f.t[i] + STEP / 2
        if runs and a - runs[-1][1] <= fill_gap:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs if b - a >= min_len]


def pad_and_merge(bursts: list, pad: float, duration: float) -> list:
    """Widen each burst by `pad` on both sides and merge overlaps."""
    out = []
    for a, b in bursts:
        a, b = max(0.0, a - pad), min(duration, b + pad)
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
