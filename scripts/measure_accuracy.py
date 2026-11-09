"""Run the controlled 17-capture experiment in BENCHMARK_HANDOFF.md."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
import uuid
from zipfile import BadZipFile

import numpy as np

import bench_lib as B
import bench_protocol as protocol
import run_bench as scoring

ROOT = Path(B.ROOT)
CACHE = ROOT / 'data' / 'bench-cache-v2'
RESULTS = ROOT / 'benchmarks'
BANK_KINDS = ('probe', 'green', 'synth')
QUERY_KINDS = ('green', 'synth', 'black')
BASELINE = 'B0 probe single, quarter-oct mean-power (current)'
PROTOTYPE = 'B1 green+synth bank, mean-dB nearest'
SELECTION = RESULTS / 'measurement_selection.json'
GATE = {'min_top3_change_pp': 10, 'confidence': 0.95, 'draws': 2000, 'seed': 0}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.part')
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def materialize(phrase, samples, settings):
    out = np.zeros(settings['buffer_samples'])
    for note in phrase['notes']:
        signal = B._note(str(Path(samples) / note['path']))[:settings['note_max_samples']]
        signal = signal * np.linspace(1, 0, len(signal)) ** 0.3
        offset = note['offset_samples']
        count = min(len(signal), len(out) - offset)
        if count > 0:
            out[offset:offset + count] += signal[:count]
    out = out[:phrase['output_samples']]
    peak = np.max(np.abs(out))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"Invalid recorded phrase {phrase['name']}")
    return (out / peak * settings['input_peak']).astype(np.float32)


def prepare(doc, samples, kinds):
    """Build only the requested inputs. Development requests omit black."""
    banks = {kind: {} for kind in BANK_KINDS}
    queries = {kind: {} for kind in kinds}
    for name, fn in B.bank_inputs('probe', samples).items():
        banks['probe'][name] = fn()
    for name, fn in B.bank_inputs('synth', samples).items():
        banks['synth'][name] = fn()
    if 'synth' in queries:
        queries['synth'] = {name: fn() for name, fn in B.query_inputs('synth', samples).items()}
    for name, phrase in doc['phrases'].items():
        instrument = phrase['instrument']
        if phrase['role'] == 'bank':
            banks[instrument][name] = materialize(phrase, samples, doc['audio'])
        elif instrument in queries:
            queries[instrument][name] = materialize(phrase, samples, doc['audio'])
    return banks, queries


def render_identity(threads):
    paths = ('scripts/bench_lib.py', 'services/audio/nam_tone_match/effects.py')
    return {'version': 2, 'source_sha256': {p: protocol.sha256_file(ROOT / p) for p in paths},
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': version('torch'),
            'neural_amp_modeler': version('neural-amp-modeler'), 'device': 'cpu', 'threads': threads}


def audio_hash(signal):
    return hashlib.sha256(np.asarray(signal, dtype='<f4').tobytes()).hexdigest()


def cache_path(capture_sha, input_sha, gain, identity, rate, peak):
    key = digest([capture_sha, input_sha, float(gain), identity, rate, peak])
    return CACHE / key[:2] / (key + '.npz')


def valid_features(values):
    return (set(values) == {'a', 'b', 'dyn', 'win'}
            and values['a'].shape == values['b'].shape == (B.NB,)
            and values['dyn'].shape == (4,)
            and values['win'].ndim == 2 and values['win'].shape[1] == B.NB
            and len(values['win']) > 0
            and all(np.isfinite(value).all() for value in values.values()))


def read_features(path):
    try:
        with np.load(path, allow_pickle=False) as npz:
            values = {key: npz[key] for key in npz.files}
        return values if valid_features(values) else None
    except (OSError, ValueError, EOFError, KeyError, BadZipFile):
        return None


def jobs(doc, banks, queries):
    """Describe each required feature with the hash of its actual input."""
    sets = [('bank', banks, doc['gains_db']['bank_common']), ('query', queries, doc['gains_db']['query'])]
    for role, groups, gains in sets:
        for kind, inputs in groups.items():
            for name, signal in inputs.items():
                input_sha = audio_hash(signal)
                for gain in gains:
                    yield role, kind, name, input_sha, gain, signal


def worker_init(doc, samples, kinds, threads):
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    global worker_jobs, worker_audio, worker_identity
    worker_jobs = list(jobs(doc, *prepare(doc, samples, kinds)))
    worker_audio = doc['audio']
    worker_identity = render_identity(threads)


def render_capture(capture):
    name, model_path, model_sha = capture
    rate = worker_audio['sample_rate']
    peak = worker_audio['output_render_peak']
    rendered = reused = 0
    for _, _, phrase, input_sha, gain, signal in worker_jobs:
        path = cache_path(model_sha, input_sha, gain, worker_identity, rate, peak)
        if read_features(path) is not None:
            reused += 1
            continue
        wet = B.render_effect(signal, rate, B._effect(model_path), gain)
        level = np.max(np.abs(wet)) if len(wet) else 0
        if not np.isfinite(level) or level <= 0:
            raise ValueError(f'{name}, {phrase}, {gain} dB: invalid or silent render')
        values = B.features(wet / level * peak)
        if not valid_features(values):
            raise ValueError(f'{name}, {phrase}, {gain} dB: invalid features')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.part')
        try:
            with temp.open('wb') as handle:
                np.savez(handle, **values)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        rendered += 1
    return name, rendered, reused


def render(doc, captures, args):
    start = time.monotonic()
    kinds = QUERY_KINDS if args.phase == 'all' else ('green', 'synth')
    count = len(captures) * (17 * 4 + len(kinds) * 10 * 3)
    print(f'Filling {count} feature entries with {args.workers} workers', flush=True)
    rendered = reused = 0
    with ProcessPoolExecutor(args.workers, initializer=worker_init,
                             initargs=(doc, str(args.samples), kinds, args.threads)) as pool:
        futures = [pool.submit(render_capture, capture) for capture in captures]
        for future in as_completed(futures):
            name, added, cached = future.result()
            rendered += added
            reused += cached
            print(f'{rendered + reused}/{count} {name}: {added} rendered, {cached} cached', flush=True)
    if rendered + reused != count:
        raise RuntimeError('Render grid is incomplete')
    save_json(ROOT / 'data' / 'bench-render-v2.json', {
        'protocol_id': doc['protocol_id'], 'expected': count, 'rendered': rendered, 'reused': reused,
        'seconds': time.monotonic() - start, 'identity': render_identity(args.threads), 'phase': args.phase,
    })


def load_grid(doc, captures, args, kinds):
    identity = render_identity(args.threads)
    requested = list(jobs(doc, *prepare(doc, args.samples, kinds)))
    banks = {kind: {} for kind in BANK_KINDS}
    queries = {kind: [] for kind in kinds}
    rate, peak = doc['audio']['sample_rate'], doc['audio']['output_render_peak']
    for name, _, model_sha in captures:
        for role, kind, phrase, input_sha, gain, _ in requested:
            path = cache_path(model_sha, input_sha, gain, identity, rate, peak)
            values = read_features(path)
            if values is None:
                raise ValueError(f'Missing or invalid features for {name}, {phrase}, {gain} dB. Run --render.')
            if role == 'bank':
                banks[kind].setdefault(name, {}).setdefault(gain, []).append(values)
            else:
                queries[kind].append({'truth': name, 'capture_sha256': model_sha, 'group': phrase,
                                      'input_sha256': input_sha, 'gain': gain, 'feats': values})
    for kind, rows in queries.items():
        if len(rows) != 510 or len({row['group'] for row in rows}) != 10:
            raise ValueError(f'Incomplete query grid for {kind}')
    return banks, queries


def method_spec(label, banks, names):
    kinds, kwargs, weighted = scoring.METHODS[label]
    bank = scoring.merge_banks(*(banks[k] for k in kinds))
    weights = scoring.estimate_weights(bank, names, kwargs.get('key', 'b')) if weighted else None
    return {'label': label, 'banks': kinds, 'parameters': kwargs,
            'weights': weights.tolist() if weights is not None else None}


def scorer(spec, banks, names):
    weights = np.asarray(spec['weights']) if spec['weights'] is not None else None
    bank = scoring.merge_banks(*(banks[k] for k in spec['banks']))
    return scoring.make_scorer(bank, names, weights=weights, **spec['parameters'])


def evaluate(score, rows, names):
    matrix = np.stack([score(row['feats']) for row in rows])
    if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
        raise ValueError('Scorer produced invalid scores')
    order = np.argsort(matrix, axis=1, kind='stable')
    truth = [names.index(row['truth']) for row in rows]
    ranks = np.argmax(order == np.asarray(truth)[:, None], axis=1) + 1
    return ranks, matrix, order[:, 0]


def metrics(ranks):
    return {'trials': len(ranks), 'top1': B.top1(ranks), 'top3': B.top3(ranks), 'mrr': B.mrr(ranks)}


def experiment_identity(doc, args):
    paths = ('scripts/measure_accuracy.py', 'scripts/bench_protocol.py', 'scripts/run_bench.py',
             'services/audio/nam_tone_match/match.py', 'services/audio/nam_tone_match/phrases.py',
             'services/audio/nam_tone_match/signals.py', 'services/audio/nam_tone_match/decode.py')
    return {'protocol_id': doc['protocol_id'], 'render': render_identity(args.threads),
            'source_sha256': {p: protocol.sha256_file(ROOT / p) for p in paths}, 'gains_db': doc['gains_db']}


def complexity(spec):
    p = spec['parameters']
    return (len(spec['banks']) + int(p.get('proto') == 'nearest') + bool(p.get('windows'))
            + bool(p.get('dyn')) + bool(p.get('alpha')))


def explore(doc, captures, args):
    if SELECTION.exists():
        raise FileExistsError('Selection already frozen. Use --final for the frozen comparison.')
    banks, queries = load_grid(doc, captures, args, ('green', 'synth'))
    names = [cap[0] for cap in captures]
    results = []
    for label in scoring.METHODS:
        spec = method_spec(label, banks, names)
        score = scorer(spec, banks, names)
        measured = {kind: metrics(evaluate(score, rows, names)[0]) for kind, rows in queries.items()}
        results.append({'method': spec, **measured})
        print(f"{label}: green top-3 {measured['green']['top3']:.2f}%", flush=True)
    challengers = (row for row in results if row['method']['label'] != BASELINE)
    chosen = max(challengers, key=lambda row: (row['green']['top3'], row['green']['mrr'],
                                               -complexity(row['method'])))['method']
    payload = {'identity': experiment_identity(doc, args), 'baseline': method_spec(BASELINE, banks, names),
               'chosen': chosen, 'fixed_prototype': method_spec(PROTOTYPE, banks, names),
               'selection_rule': 'Green top-3, green MRR, complexity, then declaration order',
               'gate': GATE, 'development': results}
    save_json(SELECTION, {**payload, 'selection_sha256': digest(payload)})
    print(f"Frozen challenger: {chosen['label']}", flush=True)


def read_selection(path, identity):
    selection = json.loads(path.read_text(encoding='utf-8'))
    checksum = selection.pop('selection_sha256')
    if checksum != digest(selection) or selection['identity'] != identity or selection['gate'] != GATE:
        raise ValueError('Frozen selection differs from the current protocol, code, or settings')
    return selection, checksum


def final(doc, captures, args):
    selection, checksum = read_selection(SELECTION, experiment_identity(doc, args))
    banks, queries = load_grid(doc, captures, args, ('black',))
    rows = queries['black']
    names = [cap[0] for cap in captures]
    groups = [row['group'] for row in rows]
    arms = {role: selection[role] for role in ('baseline', 'chosen', 'fixed_prototype')}
    computed = {role: evaluate(scorer(spec, banks, names), rows, names) for role, spec in arms.items()}
    summary = {role: {'label': arms[role]['label'], **metrics(result[0])} for role, result in computed.items()}
    base = computed['baseline'][0]
    for role in ('chosen', 'fixed_prototype'):
        delta, lo, hi = B.bootstrap_paired(base, computed[role][0], groups, B.top3,
                                            n=GATE['draws'], seed=GATE['seed'])
        summary[role].update(top3_delta_pp=delta, top3_ci95_pp=[lo, hi],
                             gate_passed=bool(delta >= GATE['min_top3_change_pp'] and lo > 0))
    by_capture = {}
    for name in names:
        mask = [row['truth'] == name for row in rows]
        by_capture[name] = {role: metrics(result[0][mask]) for role, result in computed.items()}
    trials = []
    for i, row in enumerate(rows):
        trial = {key: value for key, value in row.items() if key != 'feats'}
        trial['methods'] = {role: {'rank': int(result[0][i]), 'scores': result[1][i].tolist(),
                                   'top_capture': names[int(result[2][i])]}
                            for role, result in computed.items()}
        trials.append(trial)
    save_json(RESULTS / 'measurement_trials.json', {'selection_sha256': checksum,
                                                    'score_order': names, 'trials': trials})
    save_json(RESULTS / 'measurement_metrics.json', {'identity': selection['identity'],
              'selection_sha256': checksum, 'composite_groups': len(set(groups)),
              'uncertainty_scope': doc['uncertainty_scope'], 'gate': GATE,
              'summary': summary, 'by_capture': by_capture})
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--render', action='store_true')
    mode.add_argument('--explore', action='store_true')
    mode.add_argument('--final', action='store_true')
    parser.add_argument('--samples', type=Path, default=Path(os.environ.get('NAM_BENCH_SAMPLES') or (
        Path.home() / 'Music' / 'Sample Libraries' / 'Black and Green Guitars' / 'Samples')))
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=3)
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--phase', choices=('dev', 'all'), default='all')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.freeze:
        doc = protocol.freeze_protocol(args.samples, models_root=B.MODELS)
        print(f"Frozen protocol {doc['protocol_id']}")
        return
    doc = protocol.load_protocol(samples_root=args.samples, models_root=B.MODELS,
                                 captures_manifest=protocol.DEFAULT_CAPTURES)
    captures = [(item['name'], str(Path(B.MODELS) / item['name']), item['sha256']) for item in doc['captures']]
    if args.render:
        render(doc, captures, args)
    elif args.explore:
        explore(doc, captures, args)
    else:
        final(doc, captures, args)


if __name__ == '__main__':
    main()
