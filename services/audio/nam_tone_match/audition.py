"""Short, level-controlled audition clips for matched captures."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .decode import decode_file, decode_wav
from .effects import EffectError, _MODEL_CACHE, import_nam_models, load_effect, render_effect, resample
from .encode import write_wav

RATE = 48000
SECONDS = 8
DEFAULT_DI = Path(__file__).resolve().with_name('audition-di.wav')


def prepare_source(data_dir):
    path = os.environ.get('NAM_AUDITION_DI') or str(DEFAULT_DI)
    samples, rate = decode_file(path)
    samples = np.asarray(resample(samples[:rate * SECONDS], rate, RATE), dtype='<f4').copy()
    tail = min(len(samples), int(RATE * 0.05))
    samples[-tail:] *= np.linspace(1, 0, tail)
    peak = float(np.max(np.abs(samples), initial=0))
    if not np.isfinite(peak) or peak == 0:
        raise EffectError('The audition input must contain audible audio.')
    # Give the capture a consistent input level with headroom for loud transients.
    samples *= 0.25 / peak
    identity = hashlib.sha256(samples.tobytes()).hexdigest()
    dest = Path(data_dir) / f'audition-input-{identity}.wav'
    if not dest.is_file():
        temporary = dest.with_name(f'{dest.name}.{uuid4().hex}.part')
        try:
            write_wav(str(temporary), samples, RATE)
            os.replace(temporary, dest)
        finally:
            temporary.unlink(missing_ok=True)
    return str(dest)


def warm():
    import torch

    torch.set_num_threads(1)
    import_nam_models()


def render_audition(capture, source, compare=False):
    samples, rate = decode_wav(source)
    effect = load_effect(capture)
    if effect['kind'] == 'nam':
        import torch

        torch.set_num_threads(1)
    try:
        audio = render_effect(samples, rate, effect)
    finally:
        # The worker is long-lived and the finished WAV already serves repeated auditions.
        _MODEL_CACHE.clear()
    if not np.isfinite(audio).all():
        raise EffectError('Capture produced invalid audition audio')
    peak = float(np.max(np.abs(audio)))
    if compare:
        # Keep a shared timeline even when sample-rate conversion rounds by one frame.
        audio = np.pad(audio[:len(samples)], (0, max(0, len(samples) - len(audio))))
        dry_rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        wet_rms = np.sqrt(np.mean(audio ** 2))
        if wet_rms > 1e-8:
            level = min(dry_rms, 0.9 * wet_rms / peak)
            samples *= level / dry_rms
            audio *= level / wet_rms
        return np.column_stack((audio, samples)), rate
    if peak > 0.9:
        audio *= 0.9 / peak
    return audio, rate
