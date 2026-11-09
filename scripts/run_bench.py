"""Reproduce the original accuracy experiment. Use measure_accuracy.py for the corrected run.

  python scripts/run_bench.py --render [--samples=DIR]     fill the feature cache (slow, resumable)
  python scripts/run_bench.py --explore                    compare methods on the DEVELOPMENT queries only
  python scripts/run_bench.py --final=<method>             evaluate chosen method + baseline once on the held-out
                                                           instrument (black guitar) with paired bootstrap CIs

Methods follow ACCURACY_PLAN.md section 4 (B0 current probe score, B1 real-DI bank, B2 multi-window, B3 weighting
with an unweighted branch kept, B4 extra dynamics features).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B  # noqa: E402


def argval(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f'--{name}='):
            return a.split('=', 1)[1]
    return default


# ----------------------------------------------------------------------------- scoring
def estimate_weights(bank, caps_names, key='b', lo=0.25, hi=4.0):
    """Per-dimension sqrt(between/within) from the bank: within = across phrases at a fixed (capture, gain);
    between = across (capture, gain) centroids. Returns weights with mean 1."""
    within, cents = [], []
    for n in caps_names:
        for g, plist in bank[n].items():
            M = np.stack([p[key] for p in plist])
            within.append(M.var(0))
            cents.append(M.mean(0))
    within = np.mean(within, 0)
    between = np.var(np.stack(cents), 0)
    w = np.sqrt((between + 1e-6) / (within + 1e-6))
    w = np.clip(w / w.mean(), lo, hi)
    return w / w.mean()


def make_scorer(bank, names, key='a', proto='centroid', weights=None, alpha=0.0, dyn=0.0, windows=False):
    """Returns f(query_feats) -> scores (lower = closer) for each capture in `names`.
    alpha mixes weighted distance (alpha) with the unweighted distance (1-alpha)."""
    def vector(profile, dynamics):
        return np.concatenate([profile, dyn * dynamics]) if dyn else profile

    P = {}
    for n in names:
        P[n] = {}
        for g, plist in bank[n].items():
            vecs = np.stack([vector(p[key], p['dyn']) for p in plist])
            P[n][g] = vecs.mean(0, keepdims=True) if proto == 'centroid' else vecs
    wfull = None
    if weights is not None:
        wfull = np.concatenate([weights, np.ones(4)]) if dyn else weights

    def dist(q, P_):
        d = q[None, :] - P_
        u = np.sqrt(np.mean(d ** 2, axis=1))
        if wfull is None or alpha == 0:
            return u.min()
        wd = np.sqrt(np.mean((d * wfull[None, :]) ** 2, axis=1))
        return ((1 - alpha) * u + alpha * wd).min()

    def score(feats):
        if windows and len(feats['win']) > 1:
            qs = [vector(w, feats['dyn']) for w in feats['win']]
        else:
            qs = [vector(feats[key], feats['dyn'])]
        out = []
        for n in names:
            if len(qs) == 1:
                out.append(min(dist(qs[0], p) for p in P[n].values()))
            else:
                out.append(min(float(np.median([dist(q, p) for q in qs])) for p in P[n].values()))
        return np.array(out)
    return score


def rank_all(scorer, rows, names):
    ranks = []
    for r in rows:
        s = scorer(r['feats'])
        if s.shape != (len(names),) or not np.isfinite(s).all():
            raise ValueError(f"Invalid scores for {r['truth']}")
        order = np.argsort(s, kind='stable')
        ranks.append(1 + int(np.where(order == names.index(r['truth']))[0][0]))
    return np.array(ranks)


def line(label, ranks):
    return f'{label:58s} top1 {B.top1(ranks):5.1f}  top3 {B.top3(ranks):5.1f}  mrr {B.mrr(ranks):.3f}'


def merge_banks(*banks):
    out = {}
    for b in banks:
        for n, gd in b.items():
            for g, pl in gd.items():
                out.setdefault(n, {}).setdefault(g, []).extend(pl)
    return out


# ----------------------------------------------------------------------------- commands
def cmd_render(caps, samples):
    part = argval('part')
    if part:                       # --part=k/n : render only captures i with i % n == k (run n workers in parallel)
        k, n = (int(v) for v in part.split('/'))
        caps = [c for i, c in enumerate(caps) if i % n == k]
        print(f'worker {k}/{n}: {len(caps)} captures', flush=True)
    t = time.time()
    for kind in ('probe', 'green', 'synth'):
        B.build_bank(caps, kind, samples)
        print(f'bank {kind} done  {time.time() - t:.0f}s', flush=True)
    for kind in ('green', 'synth', 'black'):
        B.build_queries(caps, kind, samples)
        print(f'queries {kind} done  {time.time() - t:.0f}s', flush=True)


def load_all(caps, samples, query_kinds):
    banks = {k: B.build_bank(caps, k, samples) for k in ('probe', 'green', 'synth')}
    queries = {k: B.build_queries(caps, k, samples) for k in query_kinds}
    return banks, queries


METHODS = {
    # label: (bank kinds, scorer kwargs, use weights?)
    'B0 probe single, quarter-oct mean-power (current)': (['probe'], dict(key='a', proto='centroid'), False),
    'B0b probe single, mean-dB': (['probe'], dict(key='b', proto='centroid'), False),
    'B1 green bank, mean-power centroid': (['green'], dict(key='a', proto='centroid'), False),
    'B1 green bank, mean-dB centroid': (['green'], dict(key='b', proto='centroid'), False),
    'B1 green bank, mean-dB nearest-prototype': (['green'], dict(key='b', proto='nearest'), False),
    'B1 synth bank, mean-dB centroid': (['synth'], dict(key='b', proto='centroid'), False),
    'B1 green+synth bank, mean-dB centroid': (['green', 'synth'], dict(key='b', proto='centroid'), False),
    'B1 green+synth bank, mean-dB nearest': (['green', 'synth'], dict(key='b', proto='nearest'), False),
    'B1 green+synth bank, mean-power nearest': (['green', 'synth'], dict(key='a', proto='nearest'), False),
    'B1 green+synth+probe bank, mean-power nearest': (['green', 'synth', 'probe'], dict(key='a', proto='nearest'), False),
    'B1 green+synth+probe bank, mean-dB nearest': (['green', 'synth', 'probe'], dict(key='b', proto='nearest'), False),
    'B2 green+synth, multi-window median': (['green', 'synth'], dict(key='b', proto='centroid', windows=True), False),
    'B3 green+synth, weights alpha=0.5': (['green', 'synth'], dict(key='b', proto='centroid', alpha=0.5), True),
    'B3 green+synth, weights alpha=1.0': (['green', 'synth'], dict(key='b', proto='centroid', alpha=1.0), True),
    'B3+B2 green+synth, weights 0.5 + windows': (['green', 'synth'], dict(key='b', proto='centroid', alpha=0.5, windows=True), True),
    'B4 green+synth, dyn features 0.5': (['green', 'synth'], dict(key='b', proto='centroid', dyn=0.5), False),
    'B4 green+synth, dyn 0.5 + weights 0.5': (['green', 'synth'], dict(key='b', proto='centroid', alpha=0.5, dyn=0.5), True),
}


def build_method(label, banks, names):
    kinds, kw, use_w = METHODS[label]
    bank = merge_banks(*[banks[k] for k in kinds])
    weights = estimate_weights(bank, names, kw.get('key', 'b')) if use_w else None
    return make_scorer(bank, names, weights=weights, **kw)


def cmd_explore(caps, samples):
    names = [c[0] for c in caps]
    banks, queries = load_all(caps, samples, ('green', 'synth'))
    for qk in ('green', 'synth'):
        print(f'\n== DEV queries: {qk} ({len(queries[qk])} trials) ==')
        for label in METHODS:
            ranks = rank_all(build_method(label, banks, names), queries[qk], names)
            print(line(label, ranks), flush=True)


def cmd_final(caps, samples, chosen):
    names = [c[0] for c in caps]
    banks, queries = load_all(caps, samples, ('black',))
    rows = queries['black']
    groups = [r['group'] for r in rows]
    base_label = 'B0 probe single, quarter-oct mean-power (current)'
    base = rank_all(build_method(base_label, banks, names), rows, names)
    new = rank_all(build_method(chosen, banks, names), rows, names)
    print(f'\n== FINAL held-out instrument: black guitar ({len(rows)} trials, {len(set(groups))} performances) ==')
    print(line(base_label, base))
    print(line(chosen, new))
    for nm, stat in (('top1', B.top1), ('top3', B.top3), ('mrr', B.mrr)):
        d, lo, hi = B.bootstrap_paired(base, new, groups, stat)
        print(f'  paired change in {nm}: {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (bootstrap over performances)')
    per_amp = {}
    for r, rk in zip(rows, new):
        per_amp.setdefault(r['truth'], []).append(rk)
    print('  per-capture top-1 (new):')
    for n, v in per_amp.items():
        print(f'    {np.mean(np.array(v) == 1) * 100:5.1f}%  worst rank {max(v):2d}  {n[:55]}')
    return base, new


def main():
    samples = argval('samples', B.DEFAULT_SAMPLES)
    caps = B.load_baseline()
    if '--render' in sys.argv:
        cmd_render(caps, samples)
    elif '--explore' in sys.argv:
        cmd_explore(caps, samples)
    elif argval('final'):
        cmd_final(caps, samples, argval('final'))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
