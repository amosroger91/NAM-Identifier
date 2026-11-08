"""Accuracy check: does the ranker pick the right capture for a *different* performance?

Usage (from the repo root):  python scripts/accuracy_check.py [models_dir] [--samples=DIR_OF_DRY_GUITAR_WAVS]

For every capture directly in models_dir, fresh synthetic phrases are played through that capture at four
input gains. --samples=DIR adds phrases assembled from recorded WAV notes. The ranker then tries to name the
capture that produced each target. Reports aggregate top-1/top-3, MRR and confusions; exits 1 below 90% top-1.
This diagnostic does not enforce recording/session splits or test mixed/separated songs.
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
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in sys.argv[1:] if a.startswith('--'))
    samples = opts.get('samples')
    models_dir = args[0] if args else os.path.join(ROOT, 'models')
    data = os.path.join(ROOT, 'data')
    names = list_captures(models_dir)
    effects = {n: load_effect(os.path.join(models_dir, n)) for n in names}
    scenarios = []   # (label, phrase seed, drive gain in dB)
    for seed, gain in ((101, 0.0), (202, 6.0), (303, -6.0), (404, 12.0)):
        scenarios.append(('synth', seed, gain))
        if samples:
            scenarios.append(('recorded', seed + 1, gain))
    ranks, rows, confusions = [], [], {}
    print(f'{len(names)} captures, {len(scenarios)} scenarios each')
    print()
    for truth in names:
        for kind, seed, gain in scenarios:
            phrase = sample_phrase(samples, seed) if kind == 'recorded' else other_phrase(seed)
            guitar = render_effect(phrase, ANALYSIS_RATE, effects[truth], gain)
            guitar = guitar / (np.max(np.abs(guitar)) or 1.0) * 0.5   # the stem's level is arbitrary
            results = match.rank_captures(guitar, ANALYSIS_RATE, models_dir, data)
            order = [r['name'] for r in results]
            rank = order.index(truth) + 1
            ranks.append(rank)
            rows.append((truth, kind, gain, rank, results[0]['name'], results[0]['distance'],
                         results[order.index(truth)]['distance']))
            if rank > 1:
                confusions[(truth, order[0])] = confusions.get((truth, order[0]), 0) + 1
    ranks = np.array(ranks)
    for truth in names:
        mine = [r for r in rows if r[0] == truth]
        ok = sum(1 for r in mine if r[3] == 1)
        worst = max(r[3] for r in mine)
        print(f'{ok}/{len(mine)} top-1  worst rank {worst:2d}  {truth[:60]}')
    print()
    for (truth, wrong), count in sorted(confusions.items(), key=lambda kv: -kv[1])[:12]:
        print(f'  confused {count}x: {truth[:34]}  ->  {wrong[:34]}')
    n = len(ranks)
    print()
    print(f'TOP-1 {np.mean(ranks == 1) * 100:.1f}%   TOP-3 {np.mean(ranks <= 3) * 100:.1f}%   '
          f'MRR {np.mean(1.0 / ranks):.3f}   mean rank {ranks.mean():.2f} / {len(names)}   ({n} trials)')
    own = np.mean([r[6] for r in rows])
    print(f'mean distance to the true capture: {own:.2f} dB')
    return 0 if np.mean(ranks == 1) >= 0.9 else 1


if __name__ == '__main__':
    sys.exit(main())
