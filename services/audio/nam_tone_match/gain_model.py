"""Gain (distortion amount) from a rendered guitar phrase, without the dry input.

The target is the sine-probe THD that tone cards record, so a described stem and a capture's card
use one scale. The model is a ridge regression on log THD over features that depend on how the amp
shapes notes (short-term crest, sustain, spectral fill) more than on what was played. It is fit by
``scripts/fit_gain_model.py`` on many phrases and captures held out of ``scripts/tone_benchmark.py``.
"""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

RATE = 48000
FRAME = 2048
HOP = 1024
CALIBRATION_PATH = Path(__file__).with_name('gain_calibration.json')
_FREQS = np.fft.rfftfreq(FRAME, 1.0 / RATE)

FEATURE_NAMES = (
    'crest_db',            # whole-clip peak to RMS
    'short_crest_db',      # median 10 ms crest over loud passages: clipping flattens waveform peaks
    'short_crest_p90',
    'level_std_db',        # spread of frame level over loud frames: compression narrows it
    'sustain_db_s',        # median level slope after each onset: distortion holds notes up
    'flatness_mid',        # spectral flatness 300 Hz - 4 kHz: distortion fills between harmonics
    'flatness_high',       # 2 - 8 kHz
    'hf_share',            # energy above 2 kHz over energy above 80 Hz
    'tilt_std_db',         # how much the treble balance moves frame to frame
    'centroid_oct',        # spectral centroid in octaves re 1 kHz
)


def _frames(audio):
    if audio.size < FRAME:
        audio = np.pad(audio, (0, FRAME - audio.size))
    idx = np.arange(0, audio.size - FRAME + 1, HOP)
    window = np.hanning(FRAME)
    frames = np.stack([audio[i:i + FRAME] for i in idx])
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    return frames, power


def _flatness(power, low, high):
    band = (_FREQS >= low) & (_FREQS < high)
    p = power[:, band] + 1e-18
    return np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)


def features(samples):
    """Feature vector (len(FEATURE_NAMES)) or None for silence / too-short audio. Input: mono, 48 kHz."""
    audio = np.nan_to_num(np.asarray(samples, dtype=np.float64).reshape(-1))
    audio = audio - audio.mean() if audio.size else audio
    rms = math.sqrt(float(np.mean(audio ** 2))) if audio.size else 0.0
    if audio.size < RATE // 4 or rms < 1e-6:
        return None
    audio = audio / (np.max(np.abs(audio)) or 1.0)
    frames, power = _frames(audio)
    energy = power.sum(axis=1)
    level = 10 * np.log10(energy / (energy.max() + 1e-18) + 1e-12)
    loud = level > -20.0
    if loud.sum() < 3:
        return None

    # Short-term crest on 10 ms windows inside loud frames.
    win = RATE // 100
    n = audio.size // win
    blocks = audio[: n * win].reshape(n, win)
    b_rms = np.sqrt(np.mean(blocks ** 2, axis=1)) + 1e-12
    b_level = 20 * np.log10(b_rms / b_rms.max())
    use = b_level > -20.0
    crest_blocks = 20 * np.log10(np.max(np.abs(blocks[use]), axis=1) / b_rms[use])

    # Sustain: slope of frame level over the 80-400 ms after each onset.
    dt = HOP / RATE
    rise = np.diff(level, prepend=level[0])
    onsets = np.flatnonzero((rise > 4.0) & (level > -25.0))
    slopes = []
    a, b = int(0.08 / dt), int(0.4 / dt)
    for o in onsets:
        seg = level[o + a: o + b]
        if seg.size >= 4 and not np.any(np.diff(seg) > 4.0):
            slopes.append(np.polyfit(np.arange(seg.size) * dt, seg, 1)[0])
    sustain = float(np.median(slopes)) if slopes else float(np.polyfit(np.arange(level.size) * dt, level, 1)[0])
    sustain = float(np.clip(sustain, -80.0, 20.0))

    lp = power[loud]
    high = lp[:, _FREQS > 2000].sum(axis=1)
    base = lp[:, _FREQS > 80].sum(axis=1) + 1e-18
    low = lp[:, (_FREQS > 100) & (_FREQS < 800)].sum(axis=1) + 1e-18
    tilt = 10 * np.log10(high / low + 1e-12)
    mag = np.sqrt(lp.mean(axis=0))
    aud = _FREQS >= 80
    centroid = float(np.sum(_FREQS[aud] * mag[aud]) / (mag[aud].sum() + 1e-12))
    crest = 20 * math.log10(1.0 / (math.sqrt(float(np.mean(audio ** 2))) + 1e-12))
    vec = np.array([
        crest,
        float(np.median(crest_blocks)),
        float(np.percentile(crest_blocks, 90)),
        float(level[loud].std()),
        sustain,
        float(np.median(_flatness(lp, 300, 4000))),
        float(np.median(_flatness(lp, 2000, 8000))),
        float(np.median(high / base)),
        float(tilt.std()),
        math.log2(max(centroid, 50.0) / 1000.0),
    ])
    return vec if np.all(np.isfinite(vec)) else None


@lru_cache(maxsize=1)
def calibration():
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def predict_thd(samples, model=None):
    """Estimated sine-probe THD (0..~0.5) for a rendered phrase, or None."""
    model = calibration() if model is None else model
    if not model:
        return None
    vec = features(samples)
    if vec is None:
        return None
    z = (vec - np.asarray(model['mean'])) / np.asarray(model['scale'])
    log_thd = float(np.dot(z, model['coefficients']) + model['intercept'])
    return float(np.clip(math.exp(log_thd), 0.0005, 0.6))
