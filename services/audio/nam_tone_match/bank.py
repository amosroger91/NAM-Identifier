# Prototype bank: how each capture sounds with many different guitar performances
#
# Method chosen by the benchmark in ACCURACY_PLAN.md / scripts/run_bench.py ("B1 green+synth bank, mean-dB
# nearest-prototype"): render a bank of phrases through every capture at several input trims and keep the tone
# profile of each render as a prototype. A reference is scored by its distance to the NEAREST prototype of
# each capture. Matching the *kind* of guitar in the bank to the reference is what moved accuracy most.

import glob
import hashlib
import os
from typing import Dict, List, Optional

import numpy as np

from .decode import decode_wav
from .effects import EffectError, load_effect, render_effect, resample
from .phrases import training_phrases

RATE = 48000
BANK_TRIMS_DB = (-12.0, -6.0, 0.0, 6.0, 12.0, 18.0)   # digital input trim applied to the dry phrase
SYNTH_PHRASES = 8
RECORDED_PHRASES = 8
FEATURE_VERSION = 1                                    # bump when tone_profile_db or the bank recipe changes

_NOTE_CACHE: Dict[str, np.ndarray] = {}
_SYNTH_CACHE: Dict[int, list] = {}


def bank_dirs_from_env(default: Optional[str] = None) -> List[str]:
    """Directories of dry recorded-guitar clips: env NAM_DI_BANK (os.pathsep separated) or `default`."""
    raw = os.environ.get('NAM_DI_BANK') or default or ''
    return [d for d in raw.split(os.pathsep) if d and os.path.isdir(d)]


def _wav_files(dirs: List[str]) -> List[str]:
    files: List[str] = []
    for d in dirs:
        files += glob.glob(os.path.join(d, '**', '*.wav'), recursive=True)
    return sorted(set(files))


def _note(path: str) -> np.ndarray:
    if path not in _NOTE_CACHE:
        audio, rate = decode_wav(path)
        _NOTE_CACHE[path] = resample(audio, rate, RATE).astype(np.float32)
    return _NOTE_CACHE[path]


def recorded_phrase(files: List[str], seed: int, count: int = 12) -> np.ndarray:
    """One 'performance': `count` randomly chosen recorded notes laid out in time (same recipe as the benchmark)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(count, len(files)), replace=False)
    out = np.zeros(int(RATE * 12))
    pos = 0
    for i in pick:
        audio = _note(files[i])[: int(RATE * 1.6)]
        audio = audio * np.linspace(1, 0, len(audio)) ** 0.3
        n = min(len(audio), len(out) - pos)
        if n <= 0:
            break
        out[pos:pos + n] += audio[:n]
        pos += int(RATE * rng.uniform(0.5, 0.9))
    out = out[: pos + int(RATE * 1.0)]
    return (out / (np.max(np.abs(out)) or 1.0) * 0.6).astype(np.float32)


def bank_spec(di_dirs: List[str]):
    """Cheap description of the bank (no rendering): (recorded file list, info incl. bankId)."""
    files = _wav_files(di_dirs)
    spec = f'v{FEATURE_VERSION}|synth{SYNTH_PHRASES}|rec{RECORDED_PHRASES if files else 0}|' + '|'.join(
        f'{os.path.basename(f)}:{os.path.getsize(f)}' for f in files)
    return files, {'bankId': hashlib.sha1(spec.encode()).hexdigest()[:12],
                   'recordedPhrases': RECORDED_PHRASES if files else 0, 'syntheticPhrases': SYNTH_PHRASES,
                   'recordedFiles': len(files)}


def make_phrase_source(files: List[str]):
    """Callable returning the bank phrases; built once, on first use (synthesis is slow, caches make it unneeded)."""
    cache = []

    def get():
        if not cache:
            if files:
                cache.extend(recorded_phrase(files, 7001 + i) for i in range(RECORDED_PHRASES))
            if SYNTH_PHRASES not in _SYNTH_CACHE:
                _SYNTH_CACHE[SYNTH_PHRASES] = training_phrases(SYNTH_PHRASES)
            cache.extend(_SYNTH_CACHE[SYNTH_PHRASES])
        return cache
    return get


# --------------------------------------------------------------------------- feature (must match the benchmark)
FRAME, HOP, GATE_DB = 4096, 2048, 35.0
_EDGES = 90.0 * 2.0 ** (np.arange(int(np.ceil(np.log2(10000.0 / 90.0) * 4)) + 1) / 4)
_FREQS = np.fft.rfftfreq(FRAME, 1.0 / RATE)
_SEL = [(_FREQS >= _EDGES[b]) & (_FREQS < _EDGES[b + 1]) for b in range(len(_EDGES) - 1)]


def _tone_frames(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.size:
        x = x - x.mean()
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    window = np.hanning(FRAME)
    frames = np.stack([x[i:i + FRAME] * window for i in range(0, len(x) - FRAME + 1, HOP)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energy = power.sum(axis=1)
    active = energy > energy.max() * 10.0 ** (-GATE_DB / 10.0)
    return power[active], energy[active]


def _tone_bands(power):
    if not len(power):
        return np.zeros(len(_SEL))
    band = np.stack([power[:, s].mean(axis=1) for s in _SEL], axis=1)
    profile = (10.0 * np.log10(band + 1e-12)).mean(axis=0)
    return profile - profile.mean()


def tone_profile_db(samples) -> np.ndarray:
    """Mean-dB quarter-octave profile of active frames at 48 kHz, with level removed."""
    power, _ = _tone_frames(samples)
    return _tone_bands(power)


def analyze_tone(samples):
    """Return the tone profile and the dynamics used to estimate audible drive."""
    power, energy = _tone_frames(samples)
    profile = _tone_bands(power)
    if len(energy) < 3:
        return profile, None
    level = 10 * np.log10(energy / energy.max() + 1e-12)
    high = power[:, _FREQS > 2000].sum(axis=1)
    low = power[:, (_FREQS > 100) & (_FREQS < 800)].sum(axis=1)
    tilt = 10 * np.log10(high / (low + 1e-12) + 1e-12)
    x = np.asarray(samples, dtype=np.float64)
    crest = 20 * np.log10(np.abs(x).max() / (np.sqrt(np.mean(x ** 2)) + 1e-12))
    return profile, np.array([level.std(), tilt.mean(), tilt.std(), crest])


# --------------------------------------------------------------------------- index
def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def index_path(data_dir: str, capture_path: str, bank_id: str) -> str:
    return os.path.join(data_dir, 'index', f'{file_digest(capture_path)[:20]}-{bank_id}.npz')


def load_or_build_index(capture_path: str, data_dir: str, phrase_source, bank_id: str, trims=BANK_TRIMS_DB):
    """Prototypes of one capture: (profiles [n, bands], trims [n]). Cached per capture content + bank.
    Raises EffectError if the capture cannot be loaded or rendered."""
    path = index_path(data_dir, capture_path, bank_id)
    if os.path.isfile(path):
        try:
            with np.load(path) as saved:
                return saved['profiles'], saved['trims']
        except Exception:
            pass
    effect = load_effect(capture_path)
    gains = trims if effect.get('kind') == 'nam' else (0.0,)
    profiles, used = [], []
    phrases = phrase_source()
    for g in gains:
        for phrase in phrases:
            profiles.append(tone_profile_db(render_effect(phrase, RATE, effect, g)))
            used.append(g)
    profiles, used = np.array(profiles, dtype=np.float32), np.array(used, dtype=np.float32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f'.{os.getpid()}.tmp.npz'
        np.savez(tmp, profiles=profiles, trims=used)
        os.replace(tmp, path)
    except OSError:
        pass
    return profiles, used


def nearest_prototype(query_profile, profiles, trims):
    """(distance dB, trim dB) of the closest prototype."""
    d = np.sqrt(np.mean((profiles - query_profile[None, :]) ** 2, axis=1))
    i = int(np.argmin(d))
    return float(d[i]), float(trims[i])


def index_all(names: List[str], models_dir: str, data_dir: str, di_dirs: List[str], report=None, drive_model=None):
    """Build/refresh the index for the given capture file names. Returns {'indexed', 'cached', 'failed': [...]}."""
    files, info = bank_spec(di_dirs)
    phrases = make_phrase_source(files)
    out = {'indexed': 0, 'cached': 0, 'failed': [], 'bank': info}
    if drive_model is not None:
        from .drive import capture_drive
        out['driveUnmeasured'] = 0
    for name in names:
        path = os.path.join(models_dir, name)
        existed = os.path.isfile(index_path(data_dir, path, info['bankId']))
        try:
            load_or_build_index(path, data_dir, phrases, info['bankId'])
            out['cached' if existed else 'indexed'] += 1
        except EffectError as exc:
            out['failed'].append({'name': name, 'error': str(exc)})
        else:
            if drive_model is not None:
                try:
                    capture_drive(path, data_dir, drive_model)
                except (EffectError, OSError, ValueError):
                    out['driveUnmeasured'] += 1
        if report:
            report(name, out)
    return out
