"""Accuracy check: does the ranker pick the right capture for a *different* performance?

Usage (from the repo root):  python scripts/accuracy_check.py [models_dir] [--samples=DIR_OF_DRY_GUITAR_WAVS]

For every capture in models_dir (real .nam files) plus the built-in fixtures, a fresh guitar phrase (different
notes and random seed from the probe signal the matcher uses) is played through that capture at several input
gains. The ranker then has to name the capture that produced it. Prints a confusion table and exits 1 on a miss.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))

from nam_tone_match.effects import list_captures, load_effect, render_effect  # noqa: E402
from nam_tone_match.match import ANALYSIS_RATE, _pluck, tone_distance, tone_profile  # noqa: E402
from nam_tone_match import match  # noqa: E402


def other_phrase(seed, rate=ANALYSIS_RATE, seconds=9.0):
    """A phrase unlike the probe: different notes, rhythm and pluck noise."""
    rng = np.random.default_rng(seed)
    total = int(seconds * rate)
    mix = np.zeros(total)
    pos = 0
    while pos < total - rate // 2:
        note = int(rng.choice([40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64, 67, 69]))
        level = float(rng.uniform(0.3, 1.0))
        length = min(int(rng.uniform(0.3, 0.9) * rate), total - pos)
        pluck = _pluck(440.0 * 2 ** ((note - 69) / 12), length / rate, rate, rng, 0.3 + 0.6 * level)
        mix[pos:pos + length] += pluck[:length] * level
        pos += int(rng.uniform(0.2, 0.5) * rate)
    return (mix / np.max(np.abs(mix)) * 0.6).astype(np.float32)


def sample_phrase(folder, seed, rate=ANALYSIS_RATE, count=14):
    """A phrase built from real recorded guitar notes (WAV files in `folder`)."""
    import glob
    from nam_tone_match.decode import decode_wav
    from nam_tone_match.effects import resample
    files = sorted(glob.glob(os.path.join(folder, '**', '*.wav'), recursive=True))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(count, len(files)), replace=False)
    out = np.zeros(int(rate * 14))
    pos = 0
    for i in pick:
        audio, sr = decode_wav(files[i])
        audio = resample(audio, sr, rate)[: int(rate * 1.6)]
        n = min(len(audio), len(out) - pos)
        if n <= 0:
            break
        out[pos:pos + n] += audio[:n]
        pos += int(rate * rng.uniform(0.5, 0.9))
    return (out / np.max(np.abs(out)) * 0.6).astype(np.float32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    samples = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--samples=')), None)
    models_dir = args[0] if args else os.path.join(ROOT, 'models')
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ('identity', 'lowpass', 'gain'):
            with open(os.path.join(tmp, f'fixture-{kind}.nam'), 'w') as handle:
                handle.write('{"kind": "test-effect", "type": "%s"}' % kind)
        for name in list_captures(models_dir):
            with open(os.path.join(models_dir, name), 'rb') as src, open(os.path.join(tmp, name), 'wb') as dst:
                dst.write(src.read())
        names = list_captures(tmp)
        effects = {n: load_effect(os.path.join(tmp, n)) for n in names}
        data = os.path.join(tmp, 'data')
        misses = 0
        print(f'{len(names)} candidates: {", ".join(names)}\n')
        for truth in names:
            for seed, gain in ((101, 0.0), (202, 6.0), (303, -6.0)):
                phrase = sample_phrase(samples, seed) if samples else other_phrase(seed)
                guitar = render_effect(phrase, ANALYSIS_RATE, effects[truth], gain)
                guitar = guitar / (np.max(np.abs(guitar)) or 1.0) * 0.5   # the stem's level is arbitrary
                results = match.rank_captures(guitar, ANALYSIS_RATE, tmp, data)
                top = results[0]
                ok = top['name'] == truth
                misses += not ok
                second = results[1] if len(results) > 1 else {'distance': float('nan'), 'name': '-'}
                print(f"{'OK  ' if ok else 'MISS'} truth={truth[:28]:28s} gain={gain:+5.1f} -> {top['name'][:28]:28s} "
                      f"d={top['distance']:.2f} (next {second['name'][:20]} d={second['distance']:.2f}, "
                      f"chose gain {top['inputGainDb']:+.0f})")
        print(f'\n{misses} miss(es)')
        return 1 if misses else 0


if __name__ == '__main__':
    sys.exit(main())
