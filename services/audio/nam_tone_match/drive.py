"""Estimate audible drive before comparing capture tone profiles."""

from functools import lru_cache
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from . import bank
from .effects import EffectError, _MODEL_CACHE, list_captures, load_effect, render_effect, resample
from .match import ensure_clean_di

CALIBRATION_PATH = Path(__file__).with_name('drive_calibration.json')


@lru_cache(maxsize=1)
def calibration():
    return json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))


def drive_scale(distortion_db):
    return np.clip((np.asarray(distortion_db) + 32) / 16, 0, 1)


def estimate_drive(dynamics, model):
    if dynamics is None or not np.isfinite(dynamics).all():
        return None
    return float(np.clip(np.dot(dynamics, model['coefficients']) + model['intercept'], 0, 1))


def measure_distortion(capture_path):
    """Measure harmonic and intermodulation energy relative to a two-tone input."""
    effect = load_effect(capture_path)
    if effect['kind'] != 'nam':
        return np.full(len(bank.BANK_TRIMS_DB), -120.0)
    import torch

    torch.set_num_threads(1)
    count = bank.RATE // 4
    t = np.arange(2 * count) / bank.RATE
    probe = 0.1 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 332 * t))
    fundamentals = np.array([220, 332]) // 4
    products = np.array([108, 112, 440, 444, 552, 660, 664, 772, 884, 996]) // 4
    levels = []
    try:
        for trim in bank.BANK_TRIMS_DB:
            wet = render_effect(probe, bank.RATE, effect, trim)[-count:]
            power = abs(np.fft.rfft(wet, n=count)) ** 2
            base = power[fundamentals].sum()
            if not np.isfinite(power).all() or base <= 1e-12:
                raise EffectError(f'{effect["name"]}: no measurable drive probe output')
            levels.append(10 * np.log10((power[products].sum() + 1e-12) / (base + 1e-12)))
    finally:
        _MODEL_CACHE.clear()
    return np.array(levels)


def capture_drive(capture_path, data_dir, model):
    sha = bank.file_digest(capture_path)
    known = model['capture_distortion_db'].get(sha)
    if known is not None:
        return drive_scale(known)
    path = Path(data_dir) / 'drive-v1' / (sha + '.json')
    try:
        levels = np.array(json.loads(path.read_text(encoding='utf-8')), dtype=float)
        if levels.shape == (len(bank.BANK_TRIMS_DB),) and np.isfinite(levels).all():
            return drive_scale(levels)
    except (OSError, ValueError, TypeError):
        pass
    levels = measure_distortion(capture_path)
    temp = path.with_name(f'{path.name}.{uuid4().hex}.part')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps(levels.tolist()), encoding='utf-8')
        temp.replace(path)
    except OSError:
        pass
    finally:
        temp.unlink(missing_ok=True)
    return drive_scale(levels)


def compare_tone(query, profiles, trims, query_drive, drive_curve, recorded, model):
    if query_drive is None:
        return bank.nearest_prototype(query, profiles, trims)
    weights = (1 - query_drive) * np.asarray(model['clean_weights']) + query_drive * np.asarray(model['dirty_weights'])
    weight_sq = 1 - model['weight_mix'] + model['weight_mix'] * weights ** 2
    distances = np.sqrt(np.mean((profiles - query) ** 2 * weight_sq, axis=1))
    blend = model['centroid_mix'] * max(0.0, 0.7 - query_drive) if recorded else 0.0
    best = (float('inf'), 0.0)
    for trim in np.unique(trims):
        use = trims == trim
        score = distances[use].min()
        if blend:
            center = profiles[use][:recorded].mean(axis=0)
            center_dist = np.sqrt(np.mean((center - query) ** 2 * weight_sq))
            score = (1 - blend) * score + blend * center_dist
        if drive_curve is not None:
            measured = np.interp(trim, bank.BANK_TRIMS_DB, drive_curve)
            score += model['mismatch_penalty'] * max(0.0, abs(query_drive - measured) - 0.15)
        if score < best[0]:
            best = (float(score), float(trim))
    return best


def rank_captures(guitar, guitar_rate, models_dir, data_dir, report, di_dirs=None):
    """Classify the target and rank the normal capture index using that estimate."""
    model = calibration()
    ensure_clean_di(data_dir)
    di_dirs = bank.bank_dirs_from_env() if di_dirs is None else di_dirs
    files, info = bank.bank_spec(di_dirs)
    phrases = bank.make_phrase_source(files)
    query, dynamics = bank.analyze_tone(resample(guitar, guitar_rate, bank.RATE))
    drive = estimate_drive(dynamics, model)
    label = 'uncertain' if drive is None or 0.35 < drive < 0.65 else 'clean' if drive <= 0.35 else 'dirty'
    names = list_captures(models_dir)
    ranked, skipped = [], []
    unmeasured = 0
    for name in names:
        path = Path(models_dir) / name
        try:
            profiles, trims = bank.load_or_build_index(path, data_dir, phrases, info['bankId'])
        except EffectError as exc:
            skipped.append({'name': name, 'error': str(exc)})
            continue
        curve = None
        if drive is not None:
            try:
                curve = capture_drive(path, data_dir, model)
            except (EffectError, OSError, ValueError):
                unmeasured += 1
        score, trim = compare_tone(query, profiles, trims, drive, curve, info['recordedPhrases'], model)
        ranked.append({'name': name, 'distance': score, 'probeInputTrimDb': trim})
    ranked.sort(key=lambda result: (result['distance'], result['name']))
    report.update(searched=len(names), ranked=len(ranked), skipped=skipped, bank=info,
                  drive={'label': label, 'score': drive, 'unmeasuredCaptures': unmeasured})
    return ranked
