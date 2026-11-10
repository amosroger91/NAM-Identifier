"""Tone-description benchmark: gain label, adjectives and tone-search self-retrieval.

Every capture in --models is rendered through several guitar phrases, dry and with plate reverb,
and each render is described and searched exactly as the app does with an isolated stem.

  * Truth for the gain label is the capture's tone card (its sine-probe distortion reading).
  * Truth for search is the capture itself: a good search ranks it first.
  * The audition DI phrase is the one the drive model was fit on, so it is reported as "seen";
    the recorded green/black guitars and the synthetic phrases are "unseen".

Renders are cached under data/tone-bench-cache, so re-runs after a describer change only
re-describe. Usage:

    python scripts/tone_benchmark.py [--models data/test-models] [--workers 4] [--out benchmarks/tone_benchmark.json]
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import bench_lib as B  # noqa: E402
from nam_tone_match.decode import decode_file  # noqa: E402
from nam_tone_match.bank import file_digest  # noqa: E402
from nam_tone_match.effects import list_captures, load_effect, render_effect, resample  # noqa: E402

RATE = 48000
SECONDS = 8
CACHE = os.path.join(ROOT, 'data', 'tone-bench-cache')
RENDER_VERSION = 't1'
ORDER = ['clean', 'edge', 'crunch', 'overdrive', 'high_gain', 'fuzz']
CONDITIONS = {'dry': None, 'plate20': (0.2, 1.8, 11), 'plate40': (0.4, 2.6, 23)}


# ----------------------------------------------------------------------------- inputs
def phrases(samples):
    """name -> (seen, mono float32 at 48 kHz, peak 0.6)."""
    out = {}
    di, rate = decode_file(os.path.join(ROOT, 'services', 'audio', 'nam_tone_match', 'audition-di.wav'))
    di = np.asarray(resample(di, rate, RATE) if rate != RATE else di, dtype=np.float32)
    out['audition'] = (True, di)
    if os.path.isdir(samples):
        for seed in (1001, 1002, 1003):
            out[f'green{seed}'] = (False, B.recorded_phrase(samples, 'green', seed))
        for seed in (2001, 2002, 2003):
            out[f'black{seed}'] = (False, B.recorded_phrase(samples, 'black', seed))
    for seed in (3001, 3002):
        out[f'synth{seed}'] = (False, B.synth_phrase(seed))
    for name, (seen, audio) in out.items():
        audio = np.asarray(audio, dtype=np.float32)[: SECONDS * RATE]
        out[name] = (seen, audio / (np.max(np.abs(audio)) or 1.0) * 0.6)
    return out


def plate(x, mix, rt60, seed):
    """Deterministic plate-like reverb (filtered decaying noise), wet level matched to dry."""
    from scipy.signal import butter, fftconvolve, sosfilt
    rng = np.random.default_rng(seed)
    n = int(RATE * rt60 * 1.1)
    t = np.arange(n) / RATE
    ir = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
    ir = sosfilt(butter(2, 7500, 'low', fs=RATE, output='sos'), ir)
    ir = sosfilt(butter(2, 180, 'high', fs=RATE, output='sos'), ir)
    ir = np.concatenate([np.zeros(int(RATE * 0.01)), ir])
    wet = fftconvolve(x, ir)[: x.size]
    wet *= np.sqrt(np.mean(x ** 2) / (np.mean(wet ** 2) + 1e-12))
    return ((1 - mix) * x + mix * wet).astype(np.float32)


# ----------------------------------------------------------------------------- rendering (cached)
def _cache_file(sha, phrase_name, audio):
    audio_sha = hashlib.sha1(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
    key = hashlib.sha1(f'{sha}|{phrase_name}|{audio_sha}|{RENDER_VERSION}'.encode()).hexdigest()
    return os.path.join(CACHE, key[:2], key + '.npy')


def render_one(job):
    path, sha, phrase_name, audio = job
    f = _cache_file(sha, phrase_name, audio)
    if os.path.isfile(f):
        return f
    import torch
    torch.set_num_threads(1)
    y = np.asarray(render_effect(audio, RATE, load_effect(path), 0.0), dtype=np.float32)
    y = y / (np.max(np.abs(y)) or 1.0) * 0.5          # an isolated stem's level is arbitrary
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f + f'.{os.getpid()}.part.npy'
    np.save(tmp, y.astype(np.float16))
    os.replace(tmp, f)
    return f


# ----------------------------------------------------------------------------- metrics
def gain_stats(rows):
    graded = [r for r in rows if r['truth'] in ORDER]
    if not graded:
        return {}
    d = np.array([ORDER.index(r['pred']) - ORDER.index(r['truth']) if r['pred'] in ORDER else np.nan for r in graded])
    ok = ~np.isnan(d)
    return {
        'n': len(graded),
        'exact': round(float(np.mean(d[ok] == 0)) if ok.any() else 0.0, 3),
        'within1': round(float(np.mean(np.abs(d[ok]) <= 1)) if ok.any() else 0.0, 3),
        'meanDistance': round(float(np.mean(np.abs(d[ok]))) if ok.any() else 0.0, 3),
        'bias': round(float(np.mean(d[ok])) if ok.any() else 0.0, 3),
        'uncertain': int((~ok).sum()),
    }


def search_stats(rows):
    ranks = np.array([r['rank'] for r in rows])
    winners = Counter(r['top1'] for r in rows)
    top_name, top_count = winners.most_common(1)[0]
    return {
        'n': len(rows),
        'top1': round(float(np.mean(ranks == 1)), 3),
        'top5': round(float(np.mean(ranks <= 5)), 3),
        'mrr': round(float(np.mean(1.0 / ranks)), 3),
        'medianRank': float(np.median(ranks)),
        'distinctWinners': len(winners),
        'mostCommonWinner': top_name,
        'mostCommonWinnerShare': round(top_count / len(rows), 3),
    }


def adjective_stats(rows):
    n = len(rows)
    words = sorted({w for r in rows for w in r['adjectives'] + r['truthAdjectives']})
    out = {}
    for w in words:
        pred = np.array([w in r['adjectives'] for r in rows])
        true = np.array([w in r['truthAdjectives'] for r in rows])
        clap = np.array([w in r['clapAdjectives'] for r in rows])
        out[w] = {
            'rate': round(float(pred.mean()), 3),
            'truthRate': round(float(true.mean()), 3),
            'fromClap': round(float(clap.mean()), 3),
            'agreement': round(float(np.mean(pred == true)), 3),
        }
    return {'n': n, 'words': out}


def summarize(rows):
    groups = {'all': rows}
    for seen in (True, False):
        for cond in CONDITIONS:
            sel = [r for r in rows if r['seen'] == seen and r['condition'] == cond]
            if sel:
                groups[f'{"seen" if seen else "unseen"}/{cond}'] = sel
    confusion = Counter((r['truth'], r['pred']) for r in rows)
    return {
        'gain': {k: gain_stats(v) for k, v in groups.items()},
        'gainMethod': dict(Counter(r['method'] for r in rows)),
        'confusion': {t: {p: confusion[(t, p)] for p in ORDER + ['uncertain'] if confusion[(t, p)]}
                      for t in ORDER if any(confusion[(t, p)] for p in ORDER + ['uncertain'])},
        'search': {k: search_stats(v) for k, v in groups.items()},
        'adjectives': adjective_stats(rows),
        'brightnessDb': {
            'query': [round(float(q), 2) for q in np.percentile([r['brightnessDb'] for r in rows], [5, 25, 50, 75, 95])],
            'cards': [round(float(q), 2) for q in np.percentile([r['truthBrightnessDb'] for r in rows], [5, 25, 50, 75, 95])],
        },
        'tiltDb': tilt_stats(rows),
    }


def tilt_stats(rows):
    """How well a query's tilt matches its own capture's card, and the percentiles of each."""
    rows = [r for r in rows if r.get('tiltDb') is not None and r.get('truthTiltDb') is not None]
    if not rows:
        return {}
    err = np.array([r['tiltDb'] - r['truthTiltDb'] for r in rows])
    q = [r['tiltDb'] for r in rows]
    t = [r['truthTiltDb'] for r in rows]
    return {
        'query': [round(float(v), 2) for v in np.percentile(q, [5, 25, 50, 75, 95])],
        'cards': [round(float(v), 2) for v in np.percentile(t, [5, 25, 50, 75, 95])],
        'bias': round(float(err.mean()), 2),
        'absError': round(float(np.mean(np.abs(err))), 2),
        'corr': round(float(np.corrcoef(q, t)[0, 1]), 3),
    }


def print_report(s):
    print('\nGAIN LABEL (truth = sine-probe reading on the tone card)')
    print(f'  {"group":20s} {"n":>4s} {"exact":>6s} {"+-1":>6s} {"dist":>6s} {"bias":>6s} {"unc":>4s}')
    for k, g in s['gain'].items():
        if g:
            print(f'  {k:20s} {g["n"]:4d} {g["exact"]:6.2f} {g["within1"]:6.2f} {g["meanDistance"]:6.2f} {g["bias"]:+6.2f} {g["uncertain"]:4d}')
    print('  methods:', s['gainMethod'])
    print('  confusion (truth -> predicted):')
    for t, row in s['confusion'].items():
        print(f'    {t:10s}', ', '.join(f'{p}:{c}' for p, c in row.items()))
    print('\nTONE SEARCH (truth = the rendered capture itself)')
    print(f'  {"group":20s} {"n":>4s} {"top1":>6s} {"top5":>6s} {"mrr":>6s} {"winners":>8s}  most common winner')
    for k, g in s['search'].items():
        print(f'  {k:20s} {g["n"]:4d} {g["top1"]:6.2f} {g["top5"]:6.2f} {g["mrr"]:6.2f} {g["distinctWinners"]:8d}  '
              f'{g["mostCommonWinnerShare"]:.0%} {g["mostCommonWinner"][:40]}')
    print('\nADJECTIVES (rate on queries vs rate on the matching tone cards)')
    for w, a in s['adjectives']['words'].items():
        print(f'  {w:12s} query {a["rate"]:5.0%}  card {a["truthRate"]:5.0%}  from CLAP {a["fromClap"]:5.0%}  agree {a["agreement"]:5.0%}')
    b = s['brightnessDb']
    print(f'\nbrightnessDb percentiles 5/25/50/75/95  query {b["query"]}  cards {b["cards"]}')
    t = s.get('tiltDb')
    if t:
        print(f'tiltDb percentiles 5/25/50/75/95  query {t["query"]}  cards {t["cards"]}  '
              f'query-card bias {t["bias"]:+.2f}  mean abs error {t["absError"]:.2f}  corr {t["corr"]:.2f}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=os.environ.get('NAM_MODELS_DIR') or os.path.join(ROOT, 'data', 'test-models'))
    ap.add_argument('--samples', default=os.environ.get('NAM_BENCH_SAMPLES') or B.DEFAULT_SAMPLES)
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--limit', type=int, default=0, help='only the first N captures (quick check)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'benchmarks', 'tone_benchmark.json'))
    args = ap.parse_args()

    from nam_tone_match.clap_embed import get_embedder
    from nam_tone_match.tone import adjectives, describe_samples, measure
    from nam_tone_match.tone_search import load_or_build_card, search_from_description

    names = list_captures(args.models)
    if args.limit:
        names = names[: args.limit]
    inputs = phrases(args.samples)
    print(f'{len(names)} captures x {len(inputs)} phrases x {len(CONDITIONS)} conditions '
          f'({sum(1 for s, _ in inputs.values() if s)} seen phrase)', flush=True)

    caps = [(n, os.path.join(args.models, n), file_digest(os.path.join(args.models, n))) for n in names]
    jobs = [(path, sha, pn, audio) for _, path, sha in caps for pn, (_, audio) in inputs.items()]
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        files = list(pool.map(render_one, jobs, chunksize=1))
    print(f'renders ready in {time.time() - started:.0f}s', flush=True)

    embedder = get_embedder()
    print('CLAP:', 'on (' + str(embedder.model_id) + ')' if embedder.available else 'off', flush=True)
    rows = []
    file_iter = iter(files)
    for name, path, _sha in caps:
        card, _ = load_or_build_card(path, args.data, embedder)
        for pn, (seen, _audio) in inputs.items():
            rendered = np.load(next(file_iter)).astype(np.float32)
            for cond, spec in CONDITIONS.items():
                clip = rendered if spec is None else plate(rendered, *spec)
                q = describe_samples(clip, RATE, embedder=embedder)
                meas, _ = measure(clip, RATE)
                base_adj = adjectives(meas, q['gain'])
                result = search_from_description(q, args.models, args.data, embedder=embedder)
                ranking = [r['name'] for r in result['results']]
                rows.append({
                    'capture': name, 'phrase': pn, 'seen': seen, 'condition': cond,
                    'truth': card['gain']['label'], 'pred': q['gain']['label'], 'method': q['gain']['method'],
                    'adjectives': list(q['adjectives']), 'truthAdjectives': list(card['adjectives']),
                    'clapAdjectives': [w for w in q['adjectives'] if w not in base_adj],
                    'brightnessDb': float(meas['brightnessDb']),
                    'truthBrightnessDb': float(card['features']['brightnessDb']),
                    'tiltDb': float(meas['tiltDb']),
                    'truthTiltDb': card['features'].get('tiltDb'),
                    'rank': ranking.index(name) + 1 if name in ranking else len(ranking) + 1,
                    'top1': ranking[0] if ranking else '',
                })
        print(f'  described {name}', flush=True)

    summary = summarize(rows)
    print_report(summary)
    meta = {'captures': len(names), 'phrases': {k: v[0] for k, v in inputs.items()},
            'conditions': list(CONDITIONS), 'clap': bool(embedder.available)}
    rows_path = os.path.join(args.data, 'tone-bench-rows.json')
    for path, doc in ((args.out, {**meta, 'summary': summary}), (rows_path, {**meta, 'rows': rows})):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1)
            fh.write('\n')
        print('wrote', os.path.relpath(path, ROOT))


if __name__ == '__main__':
    main()
