"""Fit services/audio/nam_tone_match/gain_calibration.json (the Describe-tone gain model).

    python scripts/fit_gain_model.py --collect [--captures 64] [--workers 5]
    python scripts/fit_gain_model.py --fit [--write]

--collect renders training captures through training phrases at several input trims and stores
gain features plus the sine-probe THD target in data/gain-train.npz. Training data is kept apart
from scripts/tone_benchmark.py: captures in data/test-models are excluded, the black guitar is not
used, and green/synthetic phrases use different seeds than the benchmark.

--fit reports grouped (by capture) cross-validation for a few ridge strengths, then --write saves
the chosen fit on all rows.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import bench_lib as B  # noqa: E402
from nam_tone_match import gain_model  # noqa: E402
from nam_tone_match.bank import file_digest  # noqa: E402
from nam_tone_match.decode import decode_file  # noqa: E402
from nam_tone_match.effects import _MODEL_CACHE, list_captures, load_effect, render_effect, resample  # noqa: E402

RATE = 48000
TRIMS = (-6.0, 0.0, 6.0, 12.0)
OUT = os.path.join(ROOT, 'data', 'gain-train.npz')
CLASS_EDGES = (0.04, 0.09, 0.18, 0.30, 0.40)   # tone._THD_* thresholds
CLASSES = ('clean', 'edge', 'crunch', 'overdrive', 'high_gain', 'fuzz')


def training_phrases(samples):
    di, rate = decode_file(os.path.join(ROOT, 'services', 'audio', 'nam_tone_match', 'audition-di.wav'))
    out = {'audition': np.asarray(resample(di, rate, RATE) if rate != RATE else di, dtype=np.float32)}
    if os.path.isdir(samples):
        for seed in (7001, 7002, 7003, 7004):
            out[f'green{seed}'] = B.recorded_phrase(samples, 'green', seed)
    for i, phrase in enumerate(B.training_phrases(3)):
        out[f'synthc{i}'] = phrase
    return {k: (np.asarray(v, dtype=np.float32)[: 8 * RATE] / (np.max(np.abs(v)) or 1.0) * 0.6) for k, v in out.items()}


def probe_thd(effect, trim):
    from nam_tone_match.tone import measure
    from nam_tone_match.tone_search import _canonical
    _phrase, sine = _canonical()
    m, _ = measure(np.asarray(render_effect(sine, RATE, effect, trim), dtype=np.float64), RATE)
    return m.get('thd')


def collect_one(job):
    name, path, phrases = job
    import torch
    torch.set_num_threads(1)
    from tone_benchmark import plate
    rows = []
    try:
        effect = load_effect(path)
        if effect.get('kind') != 'nam':
            return name, rows
        for trim in TRIMS:
            thd = probe_thd(effect, trim)
            if thd is None:
                continue
            for pi, (pname, audio) in enumerate(phrases.items()):
                y = np.asarray(render_effect(audio, RATE, effect, trim), dtype=np.float32)
                y = y / (np.max(np.abs(y)) or 1.0) * 0.5
                for cond, clip in (('dry', y), ('plate', plate(y, 0.3, 2.0, 100 + pi))):
                    vec = gain_model.features(clip)
                    if vec is not None:
                        rows.append((name, pname, cond, trim, float(thd), vec))
    except Exception as exc:  # a broken capture is skipped, not fatal
        print('skip', name, exc, flush=True)
    finally:
        _MODEL_CACHE.clear()
    return name, rows


def collect(args):
    held_out = {file_digest(os.path.join(args.test_models, n)) for n in list_captures(args.test_models)}
    names = list_captures(args.models)
    random.Random(0).shuffle(names)
    phrases = training_phrases(args.samples)
    chosen = []
    for n in names:
        if len(chosen) >= args.captures:
            break
        if file_digest(os.path.join(args.models, n)) not in held_out:
            chosen.append(n)
    print(f'{len(chosen)} captures x {len(phrases)} phrases x {len(TRIMS)} trims x 2 conditions', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for name, got in pool.map(collect_one, [(n, os.path.join(args.models, n), phrases) for n in chosen]):
            rows += got
            print(f'  {name}: {len(got)} rows', flush=True)
    np.savez(OUT, capture=np.array([r[0] for r in rows]), phrase=np.array([r[1] for r in rows]),
             condition=np.array([r[2] for r in rows]), trim=np.array([r[3] for r in rows]),
             thd=np.array([r[4] for r in rows]), X=np.stack([r[5] for r in rows]))
    print('wrote', os.path.relpath(OUT, ROOT), len(rows), 'rows')


def to_class(thd):
    return np.digitize(thd, CLASS_EDGES)


def ridge_fit(X, y, w, alpha, pairs=None, invariance=0.0):
    """Weighted ridge. ``pairs`` (dry_rows, reverb_rows) adds a penalty on the prediction changing
    between a dry render and the same render with reverb, so room sound doesn't read as gain."""
    mean = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w)) + 1e-9
    z = (X - mean) / scale
    icpt = np.average(y, weights=w)
    lhs = (z.T * w) @ z + alpha * w.sum() * np.eye(X.shape[1])
    if pairs is not None and invariance:
        dry, wet = pairs
        diff = (X[wet] - X[dry]) / scale
        lhs = lhs + invariance * (diff.T * w[dry]) @ diff
    coef = np.linalg.solve(lhs, z.T @ (w * (y - icpt)))
    return {'mean': mean, 'scale': scale, 'coefficients': coef, 'intercept': float(icpt)}


def reverb_pairs(condition, capture, mask):
    """Rows are collected as (dry, plate) neighbours; keep pairs entirely inside ``mask``."""
    dry = np.flatnonzero((condition == 'dry') & mask)
    dry = dry[dry + 1 < len(condition)]
    dry = dry[(condition[dry + 1] == 'plate') & (capture[dry + 1] == capture[dry]) & mask[dry + 1]]
    return dry, dry + 1


def predict(model, X):
    z = (X - model['mean']) / model['scale']
    return np.clip(np.exp(z @ model['coefficients'] + model['intercept']), 0.0005, 0.6)


def class_weights(y_cls):
    """Balance the six classes so the dominant overdrive/high-gain rows don't swamp clean."""
    counts = np.bincount(y_cls, minlength=len(CLASSES)).astype(float)
    return 1.0 / np.maximum(counts[y_cls], 1.0)


def fit(args):
    d = np.load(OUT)
    X, thd, cap, phrase = d['X'], d['thd'], d['capture'], d['phrase']
    y, y_cls = np.log(np.clip(thd, 0.0005, 0.6)), to_class(thd)
    w = class_weights(y_cls)
    caps = np.unique(cap)
    print(f'{len(X)} rows, {len(caps)} captures; class counts',
          dict(zip(CLASSES, np.bincount(y_cls, minlength=6).tolist())))
    folds = np.array_split(np.random.default_rng(0).permutation(caps), 5)
    recorded = np.char.startswith(phrase.astype(str), 'green')
    cond = d['condition']
    best = None
    for alpha, inv in ((0.001, 0.0), (0.001, 3.0), (0.01, 3.0), (0.05, 3.0), (0.001, 10.0)):
        pred = np.zeros_like(thd)
        for held in folds:
            test = np.isin(cap, held)
            train = ~test
            idx = np.flatnonzero(train)
            pos = np.full(len(X), -1)
            pos[idx] = np.arange(idx.size)
            dry, wet = reverb_pairs(cond, cap, train)
            model = ridge_fit(X[train], y[train], w[train], alpha, (pos[dry], pos[wet]), inv)
            pred[test] = predict(model, X[test])
        dry, wet = reverb_pairs(cond, cap, np.ones(len(X), bool))
        shift = float(np.mean(np.log(pred[wet]) - np.log(pred[dry])))
        dist = np.abs(to_class(pred) - y_cls)
        bal = np.mean([np.mean(dist[y_cls == c] == 0) for c in range(6) if np.any(y_cls == c)])
        print(f'alpha {alpha:<6} invariance {inv:<4} reverb shift {shift:+.2f}  exact {np.mean(dist == 0):.2f}  within1 {np.mean(dist <= 1):.2f}  '
              f'balanced-exact {bal:.2f}  recorded-exact {np.mean(dist[recorded] == 0):.2f}  '
              f'bias {np.mean(to_class(pred) - y_cls):+.2f}')
        # Prefer reverb invariance when accuracy is within a point.
        key = round(bal, 2) - 0.02 * abs(shift) * 10
        if best is None or key > best[0]:
            best = (key, alpha, inv)
    _key, alpha, inv = best
    model = ridge_fit(X, y, w, alpha, reverb_pairs(cond, cap, np.ones(len(X), bool)), inv)
    print('chosen alpha', alpha, 'invariance', inv)
    for nm, c in sorted(zip(gain_model.FEATURE_NAMES, model['coefficients']), key=lambda t: -abs(t[1])):
        print(f'  {nm:16s} {c:+.3f}')
    if args.write:
        doc = {
            'version': 1,
            'target': 'log sine-probe THD (tone_search._canonical sine through the capture)',
            'features': list(gain_model.FEATURE_NAMES),
            'mean': model['mean'].round(6).tolist(),
            'scale': model['scale'].round(6).tolist(),
            'coefficients': model['coefficients'].round(6).tolist(),
            'intercept': round(model['intercept'], 6),
            'ridge': alpha,
            'reverb_invariance': inv,
            'training': {'rows': int(len(X)), 'captures': int(len(caps)), 'phrases': sorted(set(phrase.tolist())),
                         'trims_db': list(TRIMS), 'fitted': datetime.date.today().isoformat()},
        }
        with open(gain_model.CALIBRATION_PATH, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=2)
            fh.write('\n')
        print('wrote', os.path.relpath(gain_model.CALIBRATION_PATH, ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--collect', action='store_true')
    mode.add_argument('--fit', action='store_true')
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--models', default=os.path.join(ROOT, 'models'))
    ap.add_argument('--test-models', default=os.path.join(ROOT, 'data', 'test-models'))
    ap.add_argument('--samples', default=os.environ.get('NAM_BENCH_SAMPLES') or B.DEFAULT_SAMPLES)
    ap.add_argument('--captures', type=int, default=64)
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()
    collect(args) if args.collect else fit(args)


if __name__ == '__main__':
    main()
