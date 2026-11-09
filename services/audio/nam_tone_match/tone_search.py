"""Search a NAM library by tone.

Each capture is heard the same way: one dry plucked phrase and one sine probe
go through the file. The phrase supplies brightness, mids, and attack. The sine
supplies the gain class, because a single tone measures distortion more
repeatably than a chord. Cards are cached under ``data/tone-v1`` by capture hash.
"""

import json
import os
from functools import lru_cache
from uuid import uuid4

import numpy as np

from .bank import file_digest
from .clap_embed import NullEmbedder, get_embedder
from .effects import EffectError, list_captures, load_effect, render_effect
from .signals import pluck
from .tone import (
    ANALYSIS_RATE, adjectives, classify_dynamics, classify_harmonics, describe_samples,
    feature_vector, measure, similarity_details, to_public, cosine,
)

CARD_VERSION = 2


@lru_cache(maxsize=1)
def _canonical():
    """Deterministic dry phrase plus a mid sine. Callers must not mutate the arrays."""
    rate = ANALYSIS_RATE
    generator = np.random.default_rng(1)
    pieces = [pluck(freq, 0.45, rate, generator, brightness=0.9) for freq in (98.0, 147.0, 196.0, 294.0)]
    gap = int(0.04 * rate)
    phrase = np.zeros(sum(len(piece) for piece in pieces) + gap * (len(pieces) - 1))
    cursor = 0
    for piece in pieces:
        phrase[cursor:cursor + len(piece)] += piece
        cursor += len(piece) + gap
    peak = float(np.max(np.abs(phrase))) or 1.0
    phrase = (phrase / peak * 0.6).astype(np.float64)
    count = int(0.6 * rate)
    time = np.arange(count) / rate
    sine = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
    fade = int(0.01 * rate)
    sine[:fade] *= np.linspace(0.0, 1.0, fade)
    return phrase, sine


def _card_path(data_dir, capture_path):
    digest = file_digest(capture_path)[:20]
    return os.path.join(data_dir, 'tone-v1', digest + '.json')


def _build_card(capture_path, embedder):
    effect = load_effect(capture_path)
    phrase, sine = _canonical()
    rendered = np.asarray(render_effect(phrase, ANALYSIS_RATE, effect, 0.0), dtype=np.float64)
    probed = np.asarray(render_effect(sine, ANALYSIS_RATE, effect, 0.0), dtype=np.float64)
    color, color_audio = measure(rendered, ANALYSIS_RATE)
    probe, _probe_audio = measure(probed, ANALYSIS_RATE)
    # Similarity has to use the same measurement on the phrase that a query uses.
    # The sine probe is only the label: it is the repeatable distortion reading.
    phrase_gain = classify_harmonics(color) or classify_dynamics(color_audio)
    gain = classify_harmonics(probe) or phrase_gain
    shown = dict(color)
    shown['thd'] = probe.get('thd') if probe.get('thd') is not None else color.get('thd')
    shown['oddEven'] = probe.get('oddEven') if probe.get('oddEven') is not None else color.get('oddEven')
    embedding = None
    clap_model = None
    if embedder is not None and embedder.available:
        try:
            embedding = embedder.embed_audio(rendered, ANALYSIS_RATE)
            clap_model = embedder.model_id
        except Exception:
            embedding = None
            clap_model = None
    return {
        'version': CARD_VERSION,
        'gain': gain,
        'adjectives': adjectives(shown, gain),
        'features': {
            'brightnessDb': round(float(color['brightnessDb']), 2),
            'midDb': round(float(color['midDb']), 2),
            'crestDb': round(float(color['crestDb']), 2),
            'attack': round(float(color['attack']), 3),
            'compression': round(float(color['compression']), 3),
            'thd': None if probe.get('thd') is None else round(float(probe['thd']), 4),
            'oddEven': None if probe.get('oddEven') is None else round(float(probe['oddEven']), 3),
            'flatness': round(float(color['flatness']), 4),
            'hfShare': round(float(color['hfShare']), 4),
            'centroidHz': round(float(color['centroidHz']), 1),
        },
        'vector': feature_vector(color, phrase_gain).tolist(),
        'matchGain': phrase_gain,
        'profile': np.asarray(color['profile'], dtype=np.float64).tolist(),
        'embedding': None if embedding is None else np.asarray(embedding, dtype=np.float64).tolist(),
        'clapModel': clap_model,
    }


def _as_report(card):
    embedding = card.get('embedding')
    match_gain = card.get('matchGain') or card['gain']
    return {
        'gain': card['gain'],
        'adjectives': card['adjectives'],
        'features': card['features'],
        'match': {
            'gain': match_gain,
            'vector': np.asarray(card['vector'], dtype=np.float64),
            'profile': np.asarray(card['profile'], dtype=np.float64),
        },
        '_embedding': None if embedding is None else np.asarray(embedding, dtype=np.float64),
    }


def load_or_build_card(capture_path, data_dir, embedder):
    """Return ``(report, cached)``. Raises EffectError when the capture cannot be rendered."""
    path = _card_path(data_dir, capture_path)
    model_id = embedder.model_id if embedder is not None and embedder.available else None
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as handle:
                card = json.loads(handle.read())
            if card.get('version') == CARD_VERSION and 'vector' in card and 'profile' in card:
                if model_id is None or card.get('clapModel') == model_id:
                    return _as_report(card), True
        except (OSError, ValueError, TypeError, KeyError):
            pass
    card = _build_card(capture_path, embedder or NullEmbedder('CLAP is off.'))
    temporary = path + '.' + uuid4().hex + '.part'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(card, handle)
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        if os.path.isfile(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass
    return _as_report(card), False


def _clap_cosine(query, card):
    if query.get('_embedding') is None or card.get('_embedding') is None:
        return None
    return cosine(query['_embedding'], card['_embedding'])


def search_captures(samples, rate, models_dir, data_dir, embedder=None):
    embedder = get_embedder() if embedder is None else embedder
    query = describe_samples(samples, rate, embedder=embedder)
    return search_from_description(query, models_dir, data_dir, embedder=embedder)


def search_from_description(query, models_dir, data_dir, embedder=None):
    embedder = get_embedder() if embedder is None else embedder
    names = list_captures(models_dir)
    ranked, skipped = [], []
    cached = built = 0
    same_model = embedder.available and query.get('clap', {}).get('model') == embedder.model_id
    for name in names:
        path = os.path.join(models_dir, name)
        try:
            card, was_cached = load_or_build_card(path, data_dir, embedder)
        except EffectError as exc:
            skipped.append({'name': name, 'error': str(exc)})
            continue
        cached += int(was_cached)
        built += int(not was_cached)
        clap = _clap_cosine(query, card) if same_model else None
        similarity, parts = similarity_details(query, card['match'], clap)
        ranked.append({
            'name': name,
            'similarity': similarity,
            'scoreBreakdown': parts,
            'gain': card['gain'],
            'adjectives': card['adjectives'],
        })
    ranked.sort(key=lambda item: (-item['similarity'], item['name']))
    description = to_public(query)
    return {
        'query': description,
        'results': ranked,
        'searched': len(names),
        'ranked': len(ranked),
        'skipped': skipped,
        'cached': cached,
        'built': built,
        'clap': description['clap'],
    }


def index_captures(names, models_dir, data_dir, embedder=None):
    embedder = get_embedder() if embedder is None else embedder
    indexed = cached = 0
    failed = []
    for name in names:
        path = os.path.join(models_dir, name)
        try:
            _card, was_cached = load_or_build_card(path, data_dir, embedder)
        except EffectError as exc:
            failed.append({'name': name, 'error': str(exc)})
            continue
        cached += int(was_cached)
        indexed += int(not was_cached)
    return {'indexed': indexed, 'cached': cached, 'failed': failed, 'clap': embedder.available}
