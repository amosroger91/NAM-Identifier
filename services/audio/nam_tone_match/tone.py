"""Describe a guitar tone and score how similar two tones sound.

Measurements are ordinary DSP: spectrum, crest factor, pick attack, and harmonic
distortion. A steady tone (a sine, or a sine passed through a capture) yields a
gain class from its harmonics. A real performance usually is not a single note,
so the class then falls back to the trained dynamics model in ``drive.py``.

LAION CLAP, when enabled, adds an embedding and text-prompt scores. It does not
replace the measurements. Similarity percentages are listening scores, not a
measured accuracy.
"""

import math
import os

import numpy as np

from .effects import resample

ANALYSIS_RATE = 48000
FRAME = 4096
HOP = 2048
_FREQS = np.fft.rfftfreq(FRAME, 1.0 / ANALYSIS_RATE)
_PROFILE_EDGES = 80.0 * 2.0 ** (np.arange(0, 30) / 4.0)
_PROFILE_EDGES = _PROFILE_EDGES[_PROFILE_EDGES < 12000]

# Harmonic-distortion boundaries for a tonal excerpt. Tuned on sine, tanh,
# hard clip, and square waves; see test_tone_describe.py.
_THD_CLEAN = 0.04
_THD_EDGE = 0.09
_THD_CRUNCH = 0.18
_THD_OVERDRIVE = 0.30
_THD_HIGH_GAIN = 0.40

GAIN_NAMES = {
    'clean': 'Clean',
    'edge': 'Edge of Breakup',
    'crunch': 'Crunch',
    'overdrive': 'Overdrive',
    'high_gain': 'High Gain',
    'fuzz': 'Fuzz',
    'uncertain': 'Uncertain',
}


def _finite(value, digits=4):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _prepare(samples, rate):
    audio = np.nan_to_num(np.asarray(samples, dtype=np.float64).reshape(-1))
    if audio.size >= 2 and int(rate) != ANALYSIS_RATE:
        audio = resample(audio, float(rate), ANALYSIS_RATE)
    if audio.size > ANALYSIS_RATE * 8:
        window = ANALYSIS_RATE * 6
        hop = ANALYSIS_RATE // 2
        best_at, best_energy = 0, -1.0
        for start in range(0, audio.size - window + 1, hop):
            energy = float(np.dot(audio[start:start + window], audio[start:start + window]))
            if energy > best_energy:
                best_at, best_energy = start, energy
        audio = audio[best_at:best_at + window]
    return audio


def _spectrum(audio):
    if audio.size < FRAME:
        audio = np.pad(audio, (0, FRAME - audio.size))
    window = np.hanning(FRAME)
    frames = np.stack([audio[i:i + FRAME] * window for i in range(0, audio.size - FRAME + 1, HOP)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energy = power.sum(axis=1)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        active = np.zeros(energy.shape, dtype=bool)
    else:
        active = energy > peak * 10.0 ** (-35.0 / 10.0)
    if not np.any(active) and energy.size:
        active = energy >= peak
    return power, active


def _band_db(power, active, low, high):
    selected = (_FREQS >= low) & (_FREQS < high)
    if not np.any(active) or not np.any(selected):
        return -120.0
    mean_power = float(power[active][:, selected].mean())
    return 10.0 * math.log10(mean_power + 1e-12)


def _profile(power, active):
    mean_power = power[active].mean(axis=0) if np.any(active) else np.zeros(_FREQS.shape)
    values = []
    for low, high in zip(_PROFILE_EDGES[:-1], _PROFILE_EDGES[1:]):
        selected = (_FREQS >= low) & (_FREQS < high)
        values.append(float(mean_power[selected].mean()) if np.any(selected) else 1e-12)
    decibels = 10.0 * np.log10(np.asarray(values) + 1e-12)
    return decibels - decibels.mean()


def _attack(audio):
    win = max(1, int(ANALYSIS_RATE * 0.01))
    if audio.size < win * 4:
        return 0.0
    envelope = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode='valid'))
    step = max(1, int(ANALYSIS_RATE * 0.005))
    coarse = envelope[::step]
    if coarse.size < 4 or float(envelope.max()) <= 1e-8:
        return 0.0
    rise = np.diff(coarse)
    return float(np.clip(np.percentile(rise, 98) / (float(envelope.max()) + 1e-12), 0.0, 1.0))


def _peak_hz(segment, rate):
    segment = segment - segment.mean()
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    freqs = np.fft.rfftfreq(segment.size, 1.0 / rate)
    band = np.flatnonzero((freqs >= 70.0) & (freqs <= 1200.0))
    return float(freqs[band[int(np.argmax(spectrum[band]))]]) if band.size else None


def _single_note(audio, rate, count, fundamental_hz):
    """False when other loud windows peak at a different pitch: a phrase, not one sustained note."""
    if audio.size <= count:
        return True
    hop = max(1, count // 2)
    energies = [(float(np.dot(audio[s:s + count], audio[s:s + count])), s)
                for s in range(0, audio.size - count + 1, hop)]
    loudest = max(energy for energy, _ in energies)
    for energy, start in energies:
        if energy < 0.1 * loudest:
            continue
        other = _peak_hz(audio[start:start + count], rate)
        if other is not None and abs(other / fundamental_hz - 1.0) > 0.04:
            return False
    return True


def harmonic_stats(audio, rate):
    """THD of the loudest short window. ``tonal`` is false when no single fundamental dominates."""
    rate = int(rate)
    count = min(audio.size, int(rate * 0.35))
    if count < 64:
        return {'tonal': False, 'thd': None, 'oddEven': None, 'f0': None, 'prominence': 0.0}
    if audio.size == count:
        segment = audio
    else:
        hop = max(1, count // 4)
        best_at, best_energy = 0, -1.0
        for start in range(0, audio.size - count + 1, hop):
            energy = float(np.dot(audio[start:start + count], audio[start:start + count]))
            if energy > best_energy:
                best_at, best_energy = start, energy
        segment = audio[best_at:best_at + count]
    segment = segment - segment.mean()
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    freqs = np.fft.rfftfreq(segment.size, 1.0 / rate)
    band = np.flatnonzero((freqs >= 70.0) & (freqs <= 1200.0))
    if band.size == 0:
        return {'tonal': False, 'thd': None, 'oddEven': None, 'f0': None, 'prominence': 0.0}
    peak_index = band[int(np.argmax(spectrum[band]))]
    fundamental_hz = float(freqs[peak_index])
    half = max(float(freqs[1]) * 2.0, 6.0)

    def energy_at(freq):
        selected = (freqs >= freq - half) & (freqs <= freq + half) & (freqs < rate / 2.0 - 30.0)
        if not np.any(selected):
            return 0.0
        return float(np.sum(spectrum[selected] ** 2))

    fundamental = energy_at(fundamental_hz)
    span = (freqs >= 70.0) & (freqs <= 5000.0)
    total = float(np.sum(spectrum[span] ** 2)) + 1e-18
    prominence = fundamental / total
    odds = evens = 0.0
    harmonic_energy = 0.0
    for order in range(2, 9):
        freq = order * fundamental_hz
        if freq >= rate / 2.0 - 50.0:
            break
        partial = energy_at(freq)
        harmonic_energy += partial
        if order % 2:
            odds += partial
        else:
            evens += partial
    thd = math.sqrt(harmonic_energy / (fundamental + 1e-18))
    # Even partials of a symmetric clip are numerical dust, which would make the
    # odd/even ratio explode. Cap it so the stored feature stays readable.
    odd_even = min(30.0, math.sqrt((odds + 1e-18) / (evens + 1e-18)))
    median = float(np.median(spectrum[band])) + 1e-12
    tonal = prominence >= 0.40 and float(spectrum[peak_index]) >= 8.0 * median
    tonal = tonal and _single_note(audio, rate, count, fundamental_hz)
    return {
        'tonal': bool(tonal),
        'thd': float(thd) if tonal else None,
        'oddEven': float(odd_even) if tonal else None,
        'f0': fundamental_hz if tonal else None,
        'prominence': float(prominence),
    }


def measure(samples, rate):
    """Level-insensitive tone measurements at 48 kHz."""
    audio = _prepare(samples, rate)
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 1e-6 or audio.size < 16:
        profile = np.zeros(len(_PROFILE_EDGES) - 1)
        return {
            'brightnessDb': 0.0, 'midDb': 0.0, 'crestDb': 0.0, 'attack': 0.0,
            'compression': 0.0, 'flatness': 0.0, 'hfShare': 0.0, 'centroidHz': 0.0,
            'thd': None, 'oddEven': None, 'f0': None, 'tonal': False, 'silent': True,
            'profile': profile,
        }, audio
    power, active = _spectrum(audio)
    low = _band_db(power, active, 100.0, 400.0)
    mid = _band_db(power, active, 400.0, 2000.0)
    high = _band_db(power, active, 2000.0, 8000.0)
    mean_power = power[active].mean(axis=0) if np.any(active) else np.zeros(_FREQS.shape)
    magnitude = np.sqrt(np.maximum(mean_power, 0.0))
    audible = _FREQS >= 80.0
    weight = float(magnitude[audible].sum()) + 1e-12
    centroid = float(np.sum(_FREQS[audible] * magnitude[audible]) / weight)
    flat_band = (_FREQS >= 80.0) & (_FREQS <= 12000.0)
    flat_power = mean_power[flat_band] + 1e-18
    flatness = float(np.exp(np.mean(np.log(flat_power))) / np.mean(flat_power))
    above = float(mean_power[_FREQS >= 4000.0].sum())
    base = float(mean_power[_FREQS >= 80.0].sum()) + 1e-12
    peak = float(np.max(np.abs(audio)))
    crest = 20.0 * math.log10(peak / (rms + 1e-12))
    harmonics = harmonic_stats(audio, ANALYSIS_RATE)
    return {
        'brightnessDb': high - low,
        'midDb': mid - 0.5 * (low + high),
        'crestDb': crest,
        'attack': _attack(audio),
        'compression': float(np.clip((16.0 - crest) / 12.0, 0.0, 1.0)),
        'flatness': float(np.clip(flatness, 0.0, 1.0)),
        'hfShare': float(np.clip(above / base, 0.0, 1.0)),
        'centroidHz': centroid,
        'thd': harmonics['thd'],
        'oddEven': harmonics['oddEven'],
        'f0': harmonics['f0'],
        'tonal': harmonics['tonal'],
        'silent': False,
        'profile': _profile(power, active),
    }, audio


def _gain(label, score, confidence, method):
    return {
        'label': label,
        'name': GAIN_NAMES[label],
        'score': _finite(np.clip(score, 0.0, 1.0)),
        'confidence': _finite(np.clip(confidence, 0.0, 1.0)),
        'method': method,
    }


def classify_harmonics(meas):
    """Gain class from harmonic distortion. None when the excerpt is not a single tone."""
    if meas.get('silent') or not meas.get('tonal') or meas.get('thd') is None:
        return None
    thd = float(meas['thd'])
    score = float(np.clip(thd / 0.55, 0.0, 1.0))
    odd_even = float(meas['oddEven'] or 1.0)
    if thd < _THD_CLEAN:
        label = 'clean'
    elif thd < _THD_EDGE:
        label = 'edge'
    elif thd < _THD_CRUNCH:
        label = 'crunch'
    elif thd < _THD_OVERDRIVE:
        label = 'overdrive'
    elif thd < _THD_HIGH_GAIN or odd_even < 4.0:
        label = 'high_gain'
    else:
        label = 'fuzz'
    return _gain(label, score, 0.82, 'harmonics')


def classify_dynamics(audio):
    """Gain class from the trained clean/dirty dynamics model, split into the six tone names."""
    from .bank import analyze_tone
    from .drive import calibration, estimate_drive

    _profile, dynamics = analyze_tone(audio)
    drive = estimate_drive(dynamics, calibration())
    if drive is None:
        return _gain('uncertain', 0.5, 0.2, 'none')
    score = float(drive)
    # The same cut at 0.35 that the matcher calls clean. 0.35–0.65 was its
    # uncertain band; that span is edge, then crunch, before the dirty side.
    if score <= 0.35:
        label = 'clean'
    elif score <= 0.50:
        label = 'edge'
    elif score <= 0.65:
        label = 'crunch'
    elif score <= 0.82:
        label = 'overdrive'
    else:
        label = 'high_gain'
    return _gain(label, score, 0.48, 'dynamics')


def adjectives(meas, gain):
    """Short listening words. Bright and dark are never both returned."""
    if meas.get('silent'):
        return []
    words = []
    brightness = float(meas['brightnessDb'])
    # Guitar fundamentals sit in the low band, so a merely negative balance is not "dark".
    if brightness >= 6.0:
        words.append('bright')
    elif brightness <= -8.0:
        words.append('dark')
    if float(meas['midDb']) >= 4.0:
        words.append('mid-heavy')
    thd = float(meas['thd'] or 0.0)
    if 'bright' in words and gain['label'] in ('clean', 'edge') and thd < 0.10:
        words.append('glassy')
    # A pure sine is steady, so its crest is low. Require some distortion before
    # calling that "compressed", which is meant for a squashed guitar envelope.
    if float(meas['compression']) >= 0.72 and float(meas['attack']) <= 0.45 and thd >= 0.08:
        words.append('compressed')
    if gain['label'] in ('overdrive', 'high_gain', 'fuzz') or thd >= 0.22:
        words.append('saturated')
    return words


def feature_vector(meas, gain):
    return np.array([
        float(meas['brightnessDb']) / 15.0,
        float(meas['midDb']) / 10.0,
        float(meas['crestDb']) / 25.0,
        float(meas['attack']),
        float(gain['score']),
        float(meas['flatness']),
        float(meas['hfShare']),
    ], dtype=np.float64)


def _public_features(meas):
    return {
        'brightnessDb': _finite(meas['brightnessDb'], 2),
        'midDb': _finite(meas['midDb'], 2),
        'crestDb': _finite(meas['crestDb'], 2),
        'attack': _finite(meas['attack'], 3),
        'compression': _finite(meas['compression'], 3),
        'thd': _finite(meas['thd'], 4),
        'oddEven': _finite(meas['oddEven'], 3),
        'flatness': _finite(meas['flatness'], 4),
        'hfShare': _finite(meas['hfShare'], 4),
        'centroidHz': _finite(meas['centroidHz'], 1),
    }


def _pack(meas, gain, clap):
    return {
        'gain': gain,
        'adjectives': adjectives(meas, gain),
        'features': _public_features(meas),
        'vector': feature_vector(meas, gain),
        'profile': np.asarray(meas['profile'], dtype=np.float64),
        '_embedding': None,
        'clap': clap,
    }


def _clap_status(embedder):
    if embedder is None or not getattr(embedder, 'available', False):
        reason = getattr(embedder, 'reason', None) if embedder is not None else 'CLAP is off.'
        return {'available': False, 'model': None, 'matches': [], 'reason': reason}
    return {'available': True, 'model': embedder.model_id, 'matches': [], 'reason': None}


def describe_samples(samples, rate, embedder=None):
    """Describe one recording. ``embedder`` defaults to the process CLAP setting."""
    from .clap_embed import get_embedder

    meas, audio = measure(samples, rate)
    if meas.get('silent'):
        gain = _gain('uncertain', 0.0, 0.0, 'none')
    elif not meas.get('tonal') and float(meas['flatness']) >= 0.45:
        # Broadband noise has no pitch. The dynamics model is fit to guitar, so leave the class open.
        gain = _gain('uncertain', 0.5, 0.25, 'none')
    else:
        gain = classify_harmonics(meas) or classify_dynamics(audio)
    embedder = get_embedder() if embedder is None else embedder
    report = _pack(meas, gain, _clap_status(embedder))
    if not embedder.available or meas.get('silent'):
        return report
    try:
        embedding = embedder.embed_audio(audio, ANALYSIS_RATE)
        matches = embedder.score_prompts(embedding) if embedding is not None else []
    except Exception as exc:
        report['clap'] = {'available': False, 'model': None, 'matches': [], 'reason': f'CLAP failed: {exc}'}
        return report
    if embedding is None:
        report['clap'] = {'available': False, 'model': None, 'matches': [],
                          'reason': 'CLAP returned no embedding.'}
        return report
    report['_embedding'] = np.asarray(embedding, dtype=np.float64)
    report['clap']['matches'] = matches[:4]
    _apply_clap_label(report, meas, matches)
    return report


def _apply_clap_label(report, meas, matches):
    """Let CLAP name an uncertain tone. A harmonic measurement keeps its class."""
    gain_matches = [item for item in matches if item['group'] == 'gain']
    if report['gain']['method'] != 'harmonics' and len(gain_matches) >= 2:
        gap = gain_matches[0]['score'] - gain_matches[1]['score']
        if gap >= 0.02 and report['gain']['confidence'] < 0.55:
            label = gain_matches[0]['label']
            report['gain'] = {
                **report['gain'],
                'label': label,
                'name': GAIN_NAMES[label],
                'method': 'clap',
                'confidence': 0.4,
            }
            report['vector'] = feature_vector(meas, report['gain'])
            report['adjectives'] = adjectives(meas, report['gain'])
    brightness = float(meas['brightnessDb'])
    blocked = set()
    if brightness >= 6.0:
        blocked.add('dark')
    elif brightness <= -8.0:
        blocked.add('bright')
    if report['gain']['label'] in ('overdrive', 'high_gain', 'fuzz'):
        blocked.add('glassy')
    if 'glassy' in report['adjectives']:
        blocked.add('saturated')
    for item in matches:
        if item['group'] != 'adjective' or item['label'] in blocked:
            continue
        if item['score'] < 0.25:
            continue
        if item['label'] not in report['adjectives']:
            report['adjectives'].append(item['label'])
        break


def to_public(report):
    """JSON-ready description. Embeddings and vectors stay internal."""
    clap = dict(report.get('clap') or {})
    clap.pop('embedding', None)
    return {
        'gain': report['gain'],
        'adjectives': list(report['adjectives']),
        'features': report['features'],
        'clap': {
            'available': bool(clap.get('available')),
            'model': clap.get('model'),
            'matches': clap.get('matches') or [],
            'reason': clap.get('reason'),
        },
    }


def tone_snapshot(report):
    """Serializable measurements for reusing an isolated stem's analysis."""
    return {
        **to_public(report),
        'version': 1,
        'vector': report['vector'].tolist(),
        'profile': report['profile'].tolist(),
        '_embedding': None if report['_embedding'] is None else report['_embedding'].tolist(),
    }


def _positive_cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 1.0 if left_norm < 1e-9 and right_norm < 1e-9 else 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), 0.0, 1.0))


def _profile_similarity(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.mean(np.abs(left - right)) / 10.0, 0.0, 1.0))


def similarity_details(left, right, clap_cosine=None):
    """Return the listening score and each weighted contribution in points."""
    dsp = _positive_cosine(left['vector'], right['vector'])
    spectral = _profile_similarity(left['profile'], right['profile'])
    gain = 1.0 - abs(float(left['gain']['score']) - float(right['gain']['score']))
    if clap_cosine is None:
        weights = (0.34, 0.33, 0.33, 0.0)
        clap = 0.0
    else:
        weights = (0.22, 0.28, 0.20, 0.30)
        clap = float(np.clip(clap_cosine, 0.0, 1.0))
    confidence = min(float(left['gain']['confidence']), float(right['gain']['confidence']))
    dsp_w, gain_w, spectral_w, clap_w = weights
    if confidence < 0.45:
        shift = gain_w * 0.5
        gain_w -= shift
        dsp_w += shift * 0.5
        spectral_w += shift * 0.5
    parts = {
        'toneTraits': dsp_w * dsp,
        'drive': gain_w * gain,
        'spectrum': spectral_w * spectral,
        'embedding': None if clap_cosine is None else clap_w * clap,
    }
    total = int(np.clip(round(100.0 * sum(value for value in parts.values() if value is not None)), 0, 100))
    return total, {key: round(100.0 * value, 1) for key, value in parts.items() if value is not None}


def similarity_percent(left, right, clap_cosine=None):
    """0–100 listening similarity. Identical measurements score 100."""
    return similarity_details(left, right, clap_cosine)[0]


def compare_samples(left_samples, left_rate, right_samples, right_rate, embedder=None):
    """Similarity of two recordings, plus a description of each."""
    from .clap_embed import get_embedder

    embedder = get_embedder() if embedder is None else embedder
    left = describe_samples(left_samples, left_rate, embedder=embedder)
    right = describe_samples(right_samples, right_rate, embedder=embedder)
    clap = None
    if left.get('_embedding') is not None and right.get('_embedding') is not None:
        clap = cosine(left['_embedding'], right['_embedding'])
    return {
        'similarity': similarity_percent(left, right, clap),
        'clapCosine': None if clap is None else round(float(clap), 4),
        'left': to_public(left),
        'right': to_public(right),
    }


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def clap_requested():
    flag = os.environ.get('NAM_CLAP', '').strip().lower()
    if flag in ('0', 'off', 'false', 'no'):
        return False
    if flag in ('1', 'on', 'true', 'yes'):
        return True
    return bool(os.environ.get('NAM_CLAP_MODEL'))
