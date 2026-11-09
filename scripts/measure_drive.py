"""Calibrate and evaluate drive-aware matching against the frozen four-trim bank."""

import argparse
import json
from pathlib import Path

import numpy as np

import measure_accuracy as benchmark
from nam_tone_match import bank, drive
from nam_tone_match.decode import decode_wav
from nam_tone_match.effects import _MODEL_CACHE, load_effect, render_effect, resample

AUDITION_DI = drive.CALIBRATION_PATH.with_name('audition-di.wav')
PERFORMED_WEIGHT = 0.7


def blocks(banks, captures, curves):
    mixed = benchmark.scoring.merge_banks(banks['green'], banks['synth'])
    for name, _, sha in captures:
        for gain, features in mixed[name].items():
            label = np.interp(gain, bank.BANK_TRIMS_DB, drive.drive_scale(curves[sha]))
            yield name, gain, features, label


def drive_labels(rows, curves):
    return np.array([np.interp(r['gain'], bank.BANK_TRIMS_DB, drive.drive_scale(curves[r['capture_sha256']]))
                     for r in rows])


def fit_dynamics(groups, performed, curves, performed_weight):
    note_x = np.stack([f['dyn'] for _, _, features, _ in groups for f in features])
    note_y = np.array([label for _, _, features, label in groups for _ in features])
    real_x = np.stack([r['feats']['dyn'] for r in performed])
    real_y = drive_labels(performed, curves)
    x = np.concatenate([note_x, real_x])
    y = np.concatenate([note_y, real_y])
    # Balance clean and dirty examples within each recording domain.
    weights = []
    for part, mix in ((note_y, 1 - performed_weight), (real_y, performed_weight)):
        weights.extend(0.5 * (part / part.mean() + (1 - part) / (1 - part.mean())) * mix / len(part))
    weight = np.array(weights)
    mean = np.average(x, axis=0, weights=weight)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weight)), 1.0)
    z = (x - mean) / scale
    intercept = np.average(y, weights=weight)
    ridge = 0.04
    coeff = np.linalg.solve((z.T * weight) @ z + ridge * np.eye(x.shape[1]),
                            z.T @ (weight * (y - intercept))) / scale
    return coeff.tolist(), float(intercept - mean @ coeff)


def fit_weights(groups, clean):
    # Emphasize bands that separate the captures at this drive level across multiple phrases.
    profiles = [np.stack([f['b'] for f in features]) for _, _, features, level in groups
                if (level < 0.35 if clean else level > 0.65)]
    between = np.stack([p.mean(0) for p in profiles]).var(0)
    within = np.mean([p.var(0) for p in profiles], axis=0)
    weights = np.sqrt((between + 1e-6) / (within + 1e-6))
    weights = np.clip(weights / weights.mean(), 0.5, 2.0)
    return (weights / weights.mean()).tolist()


def calibrate(doc, banks, captures, sources):
    curves = {}
    for name, path, sha in captures:
        curves[sha] = drive.measure_distortion(path).tolist()
        print(f'Measured {name}', flush=True)
    groups = list(blocks(banks, captures, curves))
    performed = list(recorded_queries(captures, sources, calibration=True))
    coefficients, intercept = fit_dynamics(groups, performed, curves, performed_weight=PERFORMED_WEIGHT)
    model = {'protocol_id': doc['protocol_id'], 'capture_distortion_db': curves,
             'coefficients': coefficients, 'intercept': intercept,
             'clean_weights': fit_weights(groups, clean=True), 'dirty_weights': fit_weights(groups, clean=False),
             'weight_mix': 0.6, 'mismatch_penalty': 3.0, 'centroid_mix': 0.7,
             'recorded_sources': {str(path.name): benchmark.protocol.sha256_file(path) for path in sources}}
    benchmark.save_json(drive.CALIBRATION_PATH, model)
    return model


def recorded_queries(captures, sources, calibration=False):
    """Reamp two non-overlapping four-second passages from each dry recording."""
    import torch

    torch.set_num_threads(1)
    inputs = []
    for source in sources:
        audio, rate = decode_wav(str(source))
        if len(audio) < 8 * rate:
            raise ValueError(f'{source.name} must contain eight seconds of dry guitar')
        for part, signal in enumerate(np.split(audio[:8 * rate], 2)):
            signal = resample(signal, rate, bank.RATE)
            signal = signal * (0.6 / np.max(np.abs(signal)))
            gains = bank.BANK_TRIMS_DB if calibration and source == AUDITION_DI else (-6, 0, 6)
            inputs.append((source.name, part, signal, gains))
    for name, path, sha in captures:
        effect = load_effect(path)
        for group, part, signal, gains in inputs:
            for trim in gains:
                wet = render_effect(signal, bank.RATE, effect, trim)
                profile, dynamics = bank.analyze_tone(wet)
                yield {'truth': name, 'capture_sha256': sha, 'gain': trim, 'group': group, 'part': part,
                       'feats': {'b': profile, 'dyn': dynamics}}
        _MODEL_CACHE.clear()
        print(f'Reamped {name}', flush=True)


def frozen_candidates(banks, captures, model):
    for name, _, sha in captures:
        groups = list(banks['green'][name].items())
        count = len(groups[0][1])
        profiles = np.stack([f['b'] for gain, green in groups for f in green + banks['synth'][name][gain]])
        trims = np.repeat([gain for gain, _ in groups], count + len(banks['synth'][name][groups[0][0]]))
        yield name, profiles, trims, drive.drive_scale(model['capture_distortion_db'][sha]), count


def indexed_candidates(samples, data_dir, captures, model):
    """Read the production bank for the two recorded-note directories used in the benchmark."""
    _, info = bank.bank_spec([str(samples / 'green' / kind) for kind in ('ord', 'stac')])
    count = info['recordedPhrases']
    expected_trims = np.repeat(bank.BANK_TRIMS_DB, count + info['syntheticPhrases'])
    if not count:
        raise ValueError('The recorded sample bank is missing')
    for name, path, sha in captures:
        with np.load(bank.index_path(data_dir, path, info['bankId']), allow_pickle=False) as index:
            profiles, trims = index['profiles'], index['trims']
        if (not np.array_equal(trims, expected_trims) or profiles.shape != (len(trims), benchmark.B.NB)
                or not np.isfinite(profiles).all()):
            raise ValueError(f'Incomplete production index for {name}')
        yield name, profiles, trims, drive.drive_scale(model['capture_distortion_db'][sha]), count


def evaluate(candidates, rows, model):
    curves = model['capture_distortion_db']
    names = [row[0] for row in candidates]
    ranks_base, ranks_new, labels, estimates, groups = [], [], [], [], []
    for row in rows:
        features = row['feats']
        score = drive.estimate_drive(features['dyn'], model)
        labels.append(np.interp(row['gain'], bank.BANK_TRIMS_DB, drive.drive_scale(curves[row['capture_sha256']])))
        estimates.append(score)
        groups.append(row['group'])
        base = [bank.nearest_prototype(features['b'], profiles, trims)[0]
                for _, profiles, trims, _, _ in candidates]
        new = [drive.compare_tone(features['b'], profiles, trims, score, curve, count, model)[0]
               for _, profiles, trims, curve, count in candidates]
        truth = names.index(row['truth'])
        ranks_base.append(int(np.flatnonzero(np.argsort(base, kind='stable') == truth)[0] + 1))
        ranks_new.append(int(np.flatnonzero(np.argsort(new, kind='stable') == truth)[0] + 1))
    ranks_base, ranks_new = np.array(ranks_base), np.array(ranks_new)
    labels, estimates = np.array(labels), np.array(estimates, dtype=float)
    result = {}
    for key, mask in (('all', np.ones(len(labels), bool)), ('clean', labels <= 0.2), ('dirty', labels >= 0.8)):
        result[key] = {'baseline': benchmark.metrics(ranks_base[mask]), 'drive_aware': benchmark.metrics(ranks_new[mask])}
    clean, dirty = labels <= 0.2, labels >= 0.8
    result['drive_classifier'] = {
        'clean_trials': int(clean.sum()), 'dirty_trials': int(dirty.sum()),
        'clean_recall': float(np.mean(estimates[clean] <= 0.35)),
        'dirty_recall': float(np.mean(estimates[dirty] >= 0.65)),
        'ambiguous_on_extremes': float(np.mean((estimates[clean | dirty] > 0.35) & (estimates[clean | dirty] < 0.65))),
    }
    if len(set(groups)) == 10:
        result['top3_paired_change_pp'] = benchmark.B.bootstrap_paired(ranks_base, ranks_new, groups, benchmark.B.top3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('calibrate', 'dev', 'black', 'recorded'))
    parser.add_argument('--samples', type=Path,
                        default=Path.home() / 'Music' / 'Sample Libraries' / 'Black and Green Guitars' / 'Samples')
    parser.add_argument('--index-data', type=Path, help='Evaluate with the existing six-trim production index')
    parser.add_argument('--di', type=Path, action='append', default=[],
                        help='Use a performed DI WAV for calibration or recorded evaluation. Can repeat.')
    args = parser.parse_args()
    args.threads = 1
    doc = benchmark.protocol.load_protocol(samples_root=args.samples, models_root=benchmark.B.MODELS,
                                           captures_manifest=benchmark.protocol.DEFAULT_CAPTURES)
    captures = [(item['name'], str(Path(benchmark.B.MODELS) / item['name']), item['sha256']) for item in doc['captures']]
    kinds = {'dev': ('green', 'synth'), 'black': ('black',)}.get(args.phase, ())
    banks, queries = benchmark.load_grid(doc, captures, args, kinds)
    if args.phase == 'calibrate':
        if not args.di:
            parser.error('calibrate requires additional dry guitar takes supplied via --di')
        calibrate(doc, banks, captures, [AUDITION_DI, *args.di])
        return
    model = drive.calibration()
    if model['protocol_id'] != doc['protocol_id']:
        raise ValueError('Drive calibration was fitted on a different measurement protocol')
    if args.phase == 'recorded':
        queries['recorded'] = list(recorded_queries(captures, args.di or [AUDITION_DI]))
    candidates = list(indexed_candidates(args.samples, args.index_data, captures, model) if args.index_data
                      else frozen_candidates(banks, captures, model))
    print(json.dumps({kind: evaluate(candidates, rows, model) for kind, rows in queries.items()}, indent=2))


if __name__ == '__main__':
    main()
