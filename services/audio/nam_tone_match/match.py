# Tone matching utilities

import numpy as np
import os
from typing import Dict, List
from .signals import pluck as _pluck
from .effects import list_captures, load_effect, render_effect, resample, EffectError


def make_clean_di(sample_rate: int, duration: float):
    """A few plucked notes: decaying sines at 196, 247, and 294 Hz."""
    rate = int(sample_rate)
    count = int(rate * float(duration))
    if count < 1:
        return np.zeros(0, dtype=np.float32)
    timeline = np.arange(count, dtype=np.float64) / rate
    mix = np.zeros(count, dtype=np.float64)
    for index, freq in enumerate((196.0, 247.0, 294.0)):
        start = int(index * 0.35 * rate)
        if start >= count:
            break
        elapsed = timeline[: count - start]
        mix[start:] += np.sin(2.0 * np.pi * freq * elapsed) * np.exp(-4.0 * elapsed)
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.8).astype(np.float32)


def _log_magnitude(samples):
    audio = np.asarray(samples, dtype=np.float64)
    window = 1024
    hop = 256
    if audio.size < window:
        return np.log(np.abs(np.fft.rfft(audio)) + 1e-8)
    hann = np.hanning(window)
    frames = []
    for start in range(0, audio.size - window + 1, hop):
        frame = audio[start : start + window] * hann
        frames.append(np.log(np.abs(np.fft.rfft(frame)) + 1e-8))
    return np.concatenate(frames)


def _spectral_distance(target, rendered):
    count = min(len(target), len(rendered))
    if count == 0:
        return 0.0
    left = _log_magnitude(np.asarray(target, dtype=np.float64)[:count])
    right = _log_magnitude(np.asarray(rendered, dtype=np.float64)[:count])
    width = min(left.size, right.size)
    return float(np.mean((left[:width] - right[:width]) ** 2))


def rank_effects(target, clean, effects, sample_rate):
    """Rank effects by log-magnitude STFT error against `target`. Lower distance is closer.

    Each effect is a dict (a loaded capture or a bare `{'type': ...}` fixture). Ties keep name order.
    """
    heard = np.asarray(target)
    source = np.asarray(clean)
    ranked = []
    for effect in effects:
        rendered = render_effect(source, sample_rate, effect)
        ranked.append({**{k: v for k, v in effect.items() if k != 'data'},
                       "distance": _spectral_distance(heard, rendered)})
    ranked.sort(key=lambda item: (item["distance"], str(item.get("name") or "")))
    return ranked


ANALYSIS_RATE = 48000
DI_SECONDS = 2.0          # length of the spec's make_clean_di
PROBE_SECONDS = 12.0
DRIVE_GAINS_DB = (-6.0, 0.0, 6.0, 12.0)   # how hard the player hits the amp; the best fit is kept


# --------------------------------------------------------------------------- probe signal

def make_probe_di(sample_rate: int = ANALYSIS_RATE, seconds: float = PROBE_SECONDS, seed: int = 7):
    """Dry guitar-like test phrase used to excite each capture: single notes and power chords
    across the neck, soft and hard picking, so the amp's EQ, gain and compression all show."""
    rate = int(sample_rate)
    total = int(seconds * rate)
    rng = np.random.default_rng(seed)
    mix = np.zeros(total)
    phrase = [([40], 0.35), ([47], 0.35), ([52], 0.5), ([40, 47, 52], 0.7), ([45, 52, 57], 0.9),
              ([55], 0.5), ([59], 0.6), ([64], 0.7), ([67], 0.9), ([71], 1.0),
              ([43, 50, 55], 1.0), ([48, 55, 60], 0.45), ([38, 45, 50], 0.9), ([52, 59, 64], 0.6),
              ([62], 0.8), ([57], 0.4)]
    step = total / len(phrase)
    for index, (notes, level) in enumerate(phrase):
        start = int(index * step)
        length = min(int(step * 1.6), total - start)
        for k, note in enumerate(notes):
            freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
            pluck = _pluck(freq, length / rate, rate, rng, brightness=0.3 + 0.6 * level)
            mix[start:start + length] += pluck[:length] * level * (1.0 if k == 0 else 0.8)
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.6).astype(np.float32)


def ensure_probe_di(data_dir: str) -> str:
    """Write data/probe-di.wav (48 kHz) if it is not there yet and return its path."""
    from .encode import write_wav
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "probe-di.wav")
    if not os.path.isfile(path):
        write_wav(path, make_probe_di(), ANALYSIS_RATE)
    return path


def ensure_clean_di(data_dir: str) -> str:
    """Write data/clean-di.wav (48 kHz, 2 s, the three-note DI from SPEC.md) if missing."""
    from .encode import write_wav
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "clean-di.wav")
    if not os.path.isfile(path):
        write_wav(path, make_clean_di(ANALYSIS_RATE, DI_SECONDS), ANALYSIS_RATE)
    return path


def _load_probe(data_dir: str):
    from .decode import decode_wav
    ensure_clean_di(data_dir)
    return decode_wav(ensure_probe_di(data_dir))


# --------------------------------------------------------------------------- tone features

FRAME = 4096
HOP = 2048
BAND_LO, BAND_HI = 90.0, 10000.0
BANDS_PER_OCTAVE = 4
GATE_DB = 35.0            # ignore frames this far below the loudest one (silence, tails)


def _band_edges():
    count = int(np.ceil(np.log2(BAND_HI / BAND_LO) * BANDS_PER_OCTAVE))
    return BAND_LO * 2.0 ** (np.arange(count + 1) / BANDS_PER_OCTAVE)


def tone_profile(samples, rate):
    """Long-term average spectrum in dB over 1/4-octave bands, overall level removed.

    Avoids a frame-aligned comparison between different performances, while retaining sensitivity to
    their notes, pickups and playing as well as amp/cabinet coloration. Quiet frames are gated out, and
    each kept frame counts equally so loud notes do not dominate."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size < FRAME:
        audio = np.pad(audio, (0, FRAME - audio.size))
    audio = audio - audio.mean()
    window = np.hanning(FRAME)
    frames = np.stack([audio[i:i + FRAME] * window for i in range(0, audio.size - FRAME + 1, HOP)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energy = power.sum(axis=1)
    edges = _band_edges()
    if energy.max() <= 0:
        return np.zeros(len(edges) - 1)
    keep = energy > energy.max() * 10.0 ** (-GATE_DB / 10.0)
    mean_power = (power[keep] / energy[keep, None]).mean(axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    profile = np.empty(len(edges) - 1)
    for b in range(len(profile)):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        profile[b] = 10.0 * np.log10(mean_power[sel].mean() + 1e-12) if sel.any() else -120.0
    return profile - profile.mean()


def tone_distance(target_profile, rendered_profile):
    """RMS difference in dB between two tone profiles. Lower is closer; 0 is identical tone."""
    diff = np.asarray(target_profile) - np.asarray(rendered_profile)
    return float(np.sqrt(np.mean(diff ** 2)))


# --------------------------------------------------------------------------- ranking

PROFILE_VERSION = 1


def capture_profiles(path: str, data_dir: str, probe, rate: int):
    """Tone profile of the probe phrase through one capture at each drive gain: {gain: profile}.
    The probe does not depend on the guitar, so results are cached in data/profile-cache."""
    stat = os.stat(path)
    key = f"{os.path.basename(path)}-{stat.st_size}-{int(stat.st_mtime)}-v{PROFILE_VERSION}"
    key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    cache_dir = os.path.join(data_dir, "profile-cache")
    cache_file = os.path.join(cache_dir, key + ".npz")
    if os.path.isfile(cache_file):
        try:
            with np.load(cache_file) as saved:
                return {float(g): p for g, p in zip(saved["gains"], saved["profiles"])}
        except Exception:
            pass
    effect = load_effect(path)
    gains = DRIVE_GAINS_DB if effect.get("kind") == "nam" else (0.0,)
    profiles = {g: tone_profile(render_effect(probe, rate, effect, g), rate) for g in gains}
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_file, gains=np.array(list(profiles)), profiles=np.array(list(profiles.values())))
    except OSError:
        pass
    return profiles


def rank_captures(guitar, guitar_rate: int, models_dir: str, data_dir: str):
    """Rank every `.nam` in models_dir against the isolated guitar.

    Returns [{name, distance, inputGainDb}] sorted by distance ascending (ties: filename order)."""
    probe, rate = _load_probe(data_dir)
    target = tone_profile(resample(guitar, guitar_rate, rate), rate)
    ranked = []
    for name in list_captures(models_dir):
        profiles = capture_profiles(os.path.join(models_dir, name), data_dir, probe, rate)
        gain, dist = min(((g, tone_distance(target, p)) for g, p in profiles.items()), key=lambda gd: gd[1])
        ranked.append({"name": name, "distance": dist, "inputGainDb": gain})
    ranked.sort(key=lambda item: (item["distance"], item["name"]))
    return ranked


def render_capture(name: str, models_dir: str, data_dir: str, gain_db: float = 0.0):
    """Probe phrase through one capture at the given input gain. Returns (samples, sample_rate)."""
    if name != os.path.basename(name) or name in ("", ".", "..") or name not in list_captures(models_dir):
        raise EffectError(f"unknown capture {name!r}")
    probe, rate = _load_probe(data_dir)
    out = render_effect(probe, rate, load_effect(os.path.join(models_dir, name)), float(gain_db))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out, rate
