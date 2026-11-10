"""Calibrate the brightness tilt: the card offset and the bright/dark thresholds.

Tone cards hear each capture through one short synthetic phrase, which is brighter than real
playing. A query is a real recording. This script renders training captures (never the ones in
data/test-models) through the card phrase and through recorded phrases, then reports:

  * offset: how much to add to a card's tilt to put it on the recorded-phrase scale,
  * center / dark / bright: the median and the 20th/80th percentiles of captures on that scale.

The numbers are copied into tone.py (TILT_*) and tone_search.py (CARD_TILT_OFFSET). Usage:

    python scripts/fit_tilt.py [--captures 40] [--workers 4]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import bench_lib as B  # noqa: E402
from fit_gain_model import RATE, training_phrases  # noqa: E402
from nam_tone_match.bank import file_digest  # noqa: E402
from nam_tone_match.effects import list_captures, load_effect, render_effect  # noqa: E402


def tilts_for(job):
    name, path, phrases = job
    import torch
    torch.set_num_threads(1)
    from nam_tone_match.tone import measure
    from nam_tone_match.tone_search import _canonical
    try:
        effect = load_effect(path)
        if effect.get('kind') != 'nam':
            return name, None
        card_phrase, _sine = _canonical()
        card = measure(np.asarray(render_effect(card_phrase, RATE, effect, 0.0)), RATE)[0]['tiltDb']
        played = {}
        for pname, audio in phrases.items():
            played[pname] = measure(np.asarray(render_effect(audio, RATE, effect, 0.0)), RATE)[0]['tiltDb']
        return name, (card, played)
    except Exception as exc:  # a broken capture is skipped, not fatal
        print('skip', name, exc, flush=True)
        return name, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=os.path.join(ROOT, 'models'))
    ap.add_argument('--test-models', default=os.path.join(ROOT, 'data', 'test-models'))
    ap.add_argument('--samples', default=os.environ.get('NAM_BENCH_SAMPLES') or B.DEFAULT_SAMPLES)
    ap.add_argument('--captures', type=int, default=40)
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()

    held_out = {file_digest(os.path.join(args.test_models, n)) for n in list_captures(args.test_models)}
    names = list_captures(args.models)
    random.Random(1).shuffle(names)
    chosen = [n for n in names if file_digest(os.path.join(args.models, n)) not in held_out][: args.captures]
    # Recorded and DI phrases only: the synthetic training phrases are what the card already is.
    phrases = {k: v for k, v in training_phrases(args.samples).items() if not k.startswith('synth')}
    print(f'{len(chosen)} captures x {len(phrases)} recorded phrases + the card phrase', flush=True)

    cards, played = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for name, got in pool.map(tilts_for, [(n, os.path.join(args.models, n), phrases) for n in chosen]):
            if got is None:
                continue
            cards.append(got[0])
            played.append([got[1][p] for p in phrases])
    cards = np.array(cards)
    played = np.array(played)
    real = played.mean(axis=1)
    offset = float(np.mean(real - cards))
    print(f'captures used: {len(cards)}')
    print('phrase means:', {p: round(float(v), 2) for p, v in zip(phrases, played.mean(axis=0))})
    print(f'card phrase mean {cards.mean():.2f}  recorded mean {real.mean():.2f}')
    print(f'CARD_TILT_OFFSET = {offset:.2f}  (card + offset vs one recorded phrase: '
          f'sd {np.std(played - (cards + offset)[:, None]):.2f}, '
          f'corr {np.corrcoef(cards, real)[0, 1]:.2f})')
    slope, intercept = np.polyfit(cards, real, 1)
    print(f'linear map: recorded = {slope:.3f} * card + {intercept:.2f}  '
          f'(sd {np.std(played - (slope * cards + intercept)[:, None]):.2f})')
    print(f'capture spread on the recorded scale: sd {real.std():.2f}; phrase-to-phrase sd within a capture '
          f'{np.mean(played.std(axis=1)):.2f}')
    p20, p50, p80 = np.percentile(real, [20, 50, 80])
    print(f'TILT_CENTER = {p50:.1f}\nTILT_DARK = {p20:.1f}\nTILT_BRIGHT = {p80:.1f}')


if __name__ == '__main__':
    main()
