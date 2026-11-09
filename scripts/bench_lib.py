"""Benchmark library for tone-matching experiments (ACCURACY_PLAN.md, milestones A and B).

Design rules taken from the plan:
  * The 17-capture development set is frozen by content hash (benchmarks/baseline17.json).
  * Performances (phrases) and input gains are crossed independently: every phrase is rendered at every gain.
  * Candidates are characterised from a *bank* of phrases; queries are different phrases. Recorded-guitar phrases are
    split by instrument: green guitar = development, black guitar = final held-out test.
  * Statistics are bootstrapped over independent performances (phrases), never over gain variants.
Renders are expensive, so features are cached on disk keyed by capture hash + render spec.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

if os.environ.get('BENCH_THREADS'):
    import torch
    torch.set_num_threads(int(os.environ['BENCH_THREADS']))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'services', 'audio'))

from nam_tone_match.decode import decode_wav  # noqa: E402
from nam_tone_match.effects import load_effect, render_effect, resample  # noqa: E402
from nam_tone_match.match import make_probe_di  # noqa: E402
from nam_tone_match.phrases import make_phrase, training_phrases  # noqa: E402

RATE = 48000
RENDER_VERSION = 'r1'                     # bump when the renderer or conversion changes
MODELS = os.path.join(ROOT, 'models')
MANIFEST = os.path.join(ROOT, 'benchmarks', 'baseline17.json')
CACHE = os.path.join(ROOT, 'data', 'bench-cache')
DEFAULT_SAMPLES = 'C:/Users/roger/Music/Sample Libraries/Black and Green Guitars/Samples'

BANK_GAINS = (-12.0, -6.0, 0.0, 6.0, 12.0, 18.0)
QUERY_GAINS = (-6.0, 0.0, 6.0)

# ----------------------------------------------------------------------------- features
LO, HI, BPO, FRAME, GATE = 90.0, 10000.0, 4, 4096, 35.0
EDGES = LO * 2.0 ** (np.arange(int(np.ceil(np.log2(HI / LO) * BPO)) + 1) / BPO)
FREQS = np.fft.rfftfreq(FRAME, 1.0 / RATE)
SEL = [(FREQS >= EDGES[b]) & (FREQS < EDGES[b + 1]) for b in range(len(EDGES) - 1)]
NB = len(SEL)


def _frames(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    w = np.hanning(FRAME)
    idx = range(0, max(1, len(x) - FRAME + 1), FRAME // 2)
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    return np.abs(np.fft.rfft(np.stack([x[i:i + FRAME] * w for i in idx]), axis=1)) ** 2


def _profiles(p):
    """p: power frames [n, bins]. Returns (frame-normalised mean-power profile, mean-dB profile, dyn features)."""
    e = p.sum(1)
    if e.max() <= 0:
        z = np.zeros(NB)
        return z, z, np.zeros(4)
    keep = e > e.max() * 10 ** (-GATE / 10)
    pk, ek = p[keep], e[keep]
    mean_power = (pk / ek[:, None]).mean(0)
    a = np.array([10 * np.log10(mean_power[s].mean() + 1e-12) for s in SEL])
    band = np.stack([pk[:, s].mean(1) for s in SEL], 1)
    b = (10 * np.log10(band + 1e-12)).mean(0)
    lv = 10 * np.log10(ek / ek.max() + 1e-12)
    hi = pk[:, FREQS > 2000].sum(1)
    lo = pk[:, (FREQS > 100) & (FREQS < 800)].sum(1)
    tilt = 10 * np.log10(hi / (lo + 1e-12) + 1e-12)
    dyn = np.array([lv.std(), tilt.mean(), tilt.std(), 0.0])
    return a - a.mean(), b - b.mean(), dyn


def features(x):
    """Feature dict for one signal: whole-signal profiles plus 3 s windows (for multi-window scoring)."""
    x = np.asarray(x, np.float32)
    p = _frames(x)
    a, b, dyn = _profiles(p)
    wins = []
    step = int(1.5 * RATE)
    for s in range(0, max(1, len(x) - 3 * RATE + 1), step):
        _, wb, _ = _profiles(_frames(x[s:s + 3 * RATE]))
        wins.append(wb)
    xs = np.asarray(x, np.float64)
    dyn[3] = 20 * np.log10(np.abs(xs).max() / (np.sqrt(np.mean(xs ** 2)) + 1e-12))   # crest factor
    return {'a': a.astype(np.float32), 'b': b.astype(np.float32), 'dyn': dyn.astype(np.float32),
            'win': np.stack(wins).astype(np.float32)}


# ----------------------------------------------------------------------------- frozen capture set
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_baseline(strict=True):
    """Return [(name, path, sha)] for the frozen set, verifying content hashes."""
    manifest = json.load(open(MANIFEST, encoding='utf-8'))
    out = []
    for item in manifest['captures']:
        path = os.path.join(MODELS, item['name'])
        if not os.path.isfile(path):
            raise SystemExit(f"frozen capture missing: {item['name']}")
        if strict and sha256(path) != item['sha256']:
            raise SystemExit(f"frozen capture changed on disk: {item['name']}")
        out.append((item['name'], path, item['sha256']))
    return out


# ----------------------------------------------------------------------------- phrases
_NOTE_CACHE = {}


def _note_files(samples, instrument):
    files = []
    for art in ('ord', 'stac'):
        files += glob.glob(os.path.join(samples, instrument, art, '*.wav'))
    return sorted(files)


def _note(path):
    if path not in _NOTE_CACHE:
        a, sr = decode_wav(path)
        _NOTE_CACHE[path] = resample(a, sr, RATE).astype(np.float32)
    return _NOTE_CACHE[path]


def recorded_phrase(samples, instrument, seed, count=12):
    """Phrase assembled from recorded notes of one instrument (independent performance per seed)."""
    files = _note_files(samples, instrument)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=count, replace=False)
    out = np.zeros(int(RATE * 12))
    pos = 0
    for i in pick:
        a = _note(files[i])[: int(RATE * 1.6)]
        a = a * np.linspace(1, 0, len(a)) ** 0.3          # avoid hard truncation clicks
        n = min(len(a), len(out) - pos)
        if n <= 0:
            break
        out[pos:pos + n] += a[:n]
        pos += int(RATE * rng.uniform(0.5, 0.9))
    out = out[: pos + int(RATE * 1.0)]
    return (out / np.max(np.abs(out)) * 0.6).astype(np.float32)


def synth_phrase(seed):
    return make_phrase(seed, 9.0)


# ----------------------------------------------------------------------------- cached rendering
_EFFECTS = {}


def _effect(path):
    if path not in _EFFECTS:
        _EFFECTS[path] = load_effect(path)
    return _EFFECTS[path]


def rendered_features(cap, spec, make_input, gain, normalize_peak=0.5):
    """Features of `make_input()` through capture `cap` at `gain` dB, cached by (capture hash, spec, gain)."""
    name, path, sha = cap
    key = hashlib.sha1(f'{sha}|{spec}|{gain}|{RENDER_VERSION}'.encode()).hexdigest()
    f = os.path.join(CACHE, key[:2], key + '.npz')
    if os.path.isfile(f):
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    y = render_effect(make_input(), RATE, _effect(path), gain)
    y = y / (np.abs(y).max() or 1.0) * normalize_peak       # a stem's absolute level is arbitrary
    feats = features(y)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **feats)
    return feats


# ----------------------------------------------------------------------------- bank / query sets
def bank_inputs(kind, samples):
    """name -> input factory for the candidate-side bank."""
    if kind == 'probe':
        return {'probe': lambda: make_probe_di(RATE, 12.0)}
    if kind == 'green':
        return {f'green{s}': (lambda s=s: recorded_phrase(samples, 'green', s)) for s in range(7001, 7009)}
    if kind == 'synth':
        ph = training_phrases(8)
        return {f'synth{i}': (lambda p=p: p) for i, p in enumerate(ph)}
    raise ValueError(kind)


def query_inputs(kind, samples, count=10):
    """Held-out performances. Seeds are disjoint from the bank seeds."""
    if kind == 'green':
        return {f'q-green{s}': (lambda s=s: recorded_phrase(samples, 'green', s)) for s in range(1001, 1001 + count)}
    if kind == 'black':
        return {f'q-black{s}': (lambda s=s: recorded_phrase(samples, 'black', s)) for s in range(2001, 2001 + count)}
    if kind == 'synth':
        return {f'q-synth{s}': (lambda s=s: synth_phrase(s)) for s in range(3001, 3001 + count)}
    raise ValueError(kind)


def build_bank(caps, kind, samples, gains=BANK_GAINS):
    """{capture: {gain: [feature dict per bank phrase]}}"""
    inputs = bank_inputs(kind, samples)
    gains = (-6.0, 0.0, 6.0, 12.0) if kind == 'probe' else gains
    out = {}
    for cap in caps:
        out[cap[0]] = {g: [rendered_features(cap, f'bank:{n}', fn, g) for n, fn in inputs.items()] for g in gains}
    return out


def build_queries(caps, kind, samples, count=10, gains=QUERY_GAINS):
    """list of dicts {truth, group (performance id), gain, feats}"""
    inputs = query_inputs(kind, samples, count)
    rows = []
    for cap in caps:
        for n, fn in inputs.items():
            for g in gains:
                rows.append({'truth': cap[0], 'group': n, 'gain': g, 'feats': rendered_features(cap, f'query:{n}', fn, g)})
    return rows


# ----------------------------------------------------------------------------- statistics
def bootstrap_paired(ranks_a, ranks_b, groups, stat, n=2000, seed=0):
    """Paired bootstrap over independent performances. Returns (mean diff, lo, hi) of stat(b)-stat(a)."""
    ra, rb, groups = np.asarray(ranks_a), np.asarray(ranks_b), np.asarray(groups)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        diffs.append(stat(rb[idx]) - stat(ra[idx]))
    point = stat(rb) - stat(ra)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def top1(r):
    return float(np.mean(np.asarray(r) == 1) * 100)


def top3(r):
    return float(np.mean(np.asarray(r) <= 3) * 100)


def mrr(r):
    return float(np.mean(1.0 / np.asarray(r)))
