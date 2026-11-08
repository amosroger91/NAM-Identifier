# Loading and rendering "effects": real NAM captures and small test fixtures

import json
import os
from typing import Any, Dict, List

import numpy as np

NAM_REQUIRED_KEYS = ("version", "architecture", "config", "weights")
FIXTURE_TYPES = ("identity", "lowpass", "gain")


class EffectError(ValueError):
    """A capture file could not be loaded or rendered."""


def apply_lowpass_filter(audio: np.ndarray, sample_rate: float, cutoff_freq: float = 1000.0) -> np.ndarray:
    """Short moving-average lowpass."""
    audio = np.asarray(audio)
    if len(audio) > 1:
        window_size = max(1, int(sample_rate / cutoff_freq))
        if window_size < len(audio):
            return np.convolve(audio, np.ones(window_size) / window_size, mode='same')
    return audio.copy()


def apply_highpass_filter(audio: np.ndarray, sample_rate: float, cutoff_freq: float = 100.0) -> np.ndarray:
    """Original minus a moving-average lowpass."""
    audio = np.asarray(audio)
    return audio - apply_lowpass_filter(audio, sample_rate, cutoff_freq)


def list_captures(models_dir: str) -> List[str]:
    """File names of every `.nam` capture in models_dir, in filename order."""
    if not models_dir or not os.path.isdir(models_dir):
        return []
    return sorted(n for n in os.listdir(models_dir)
                  if n.lower().endswith('.nam') and os.path.isfile(os.path.join(models_dir, n)))


def load_effect(path: str) -> Dict[str, Any]:
    """Load one `.nam` file. Returns a dict with `name`, `kind` ('test-effect' or 'nam') and the parsed data."""
    name = os.path.basename(path)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise EffectError(f"{name}: cannot read capture ({exc})")
    if not isinstance(data, dict):
        raise EffectError(f"{name}: capture must be a JSON object")
    if data.get('kind') == 'test-effect':
        if data.get('type') not in FIXTURE_TYPES:
            raise EffectError(f"{name}: unknown test-effect type {data.get('type')!r}")
        return {'name': name, 'kind': 'test-effect', 'type': data['type']}
    data = upgrade_legacy_wavenet(unwrap_slimmable(data, name))
    missing = [key for key in NAM_REQUIRED_KEYS if key not in data]
    if missing:
        raise EffectError(f"{name}: not a valid NAM capture, missing {', '.join(repr(k) for k in missing)}")
    return {'name': name, 'kind': 'nam', 'data': data,
            'sample_rate': float(data.get('sample_rate') or 48000)}


def unwrap_slimmable(data: Dict[str, Any], name: str = 'capture') -> Dict[str, Any]:
    """Newer captures use a `SlimmableContainer` of several sub-models of growing size.
    Use the biggest one (highest `max_value`), which is the full-quality model."""
    if data.get('architecture') != 'SlimmableContainer':
        return data
    subs = (data.get('config') or {}).get('submodels') or []
    subs = [s for s in subs if isinstance(s, dict) and isinstance(s.get('model'), dict)]
    if not subs:
        raise EffectError(f"{name}: SlimmableContainer has no usable submodels")
    best = dict(max(subs, key=lambda s: s.get('max_value', 0))['model'])
    best.setdefault('sample_rate', data.get('sample_rate'))
    best.setdefault('metadata', data.get('metadata'))
    return best


def upgrade_legacy_wavenet(data: Dict[str, Any]) -> Dict[str, Any]:
    """Version 0.5.x WaveNet captures (the bulk of community files) describe each layer array with
    `head_size`/`head_bias`/`gated` and a plain activation name. Current `nam` only accepts the newer
    layout (`head` object, per-layer activation and gating lists), so rewrite the config. The weight
    vector layout is retained, and `nam` rejects a wrong weight count. Renderer conformance still needs
    an audio comparison: a valid weight count alone does not prove equivalent processing."""
    if data.get('architecture') != 'WaveNet':
        return data
    layers = (data.get('config') or {}).get('layers')
    if not layers or all('head' in lc for lc in layers):
        return data
    upgraded = []
    for lc in layers:
        lc = dict(lc)
        count = len(lc.get('dilations') or [])
        if 'head' not in lc and 'head_size' in lc:
            lc['head'] = {'out_channels': lc.pop('head_size'), 'kernel_size': 1,
                          'bias': bool(lc.pop('head_bias', False))}
        activation = lc.get('activation')
        if isinstance(activation, str):
            lc['activation'] = [{'type': activation}] * count
        if 'gated' in lc:
            gated = bool(lc.pop('gated'))
            lc['gating_mode'] = ['gated' if gated else 'none'] * count
            lc['secondary_activation'] = [{'type': 'Sigmoid'} if gated else None] * count
        upgraded.append(lc)
    config = dict(data['config'])
    config['layers'] = upgraded
    return {**data, 'config': config}


def render_fixture(clean, effect) -> np.ndarray:
    audio = np.asarray(clean, dtype=np.float64)
    kind = effect.get('type')
    if kind == 'lowpass':
        return np.convolve(audio, np.ones(8) / 8.0, mode='same')
    if kind == 'gain':
        return audio * 0.5
    return np.array(audio, copy=True)


def resample(audio, from_rate: float, to_rate: float) -> np.ndarray:
    """Band-limited (FFT) resample: no imaging or droop, unlike linear interpolation."""
    audio = np.asarray(audio, dtype=np.float64)
    if int(from_rate) == int(to_rate) or audio.size < 2:
        return audio
    n = audio.size
    m = max(1, int(round(n * float(to_rate) / float(from_rate))))
    spectrum = np.fft.rfft(audio)
    wanted = m // 2 + 1
    if wanted > spectrum.size:
        spectrum = np.concatenate([spectrum, np.zeros(wanted - spectrum.size, dtype=spectrum.dtype)])
    else:
        spectrum = spectrum[:wanted]
    return np.fft.irfft(spectrum, m) * (m / n)


def import_nam_models():
    """Return nam's `init_from_nam`.

    A plain `import nam` also pulls in the training stack (scipy, librosa, pytorch-lightning), which
    some machines block or lack. Inference only needs `nam.models`, so on failure we load it without
    the package `__init__` and without the training-only modules."""
    try:
        from nam.models import init_from_nam
        return init_from_nam
    except Exception:
        pass
    import importlib.util
    import sys
    import types
    for key in [k for k in sys.modules if k == 'nam' or k.startswith('nam.')]:
        del sys.modules[key]
    spec = importlib.util.find_spec('nam')
    if spec is None or not spec.origin:
        raise ImportError("the 'neural-amp-modeler' package is not installed")
    pkg = types.ModuleType('nam')
    pkg.__path__ = [os.path.dirname(spec.origin)]
    sys.modules['nam'] = pkg
    for stub in ('nam.data', 'nam.models.losses'):
        sys.modules[stub] = types.ModuleType(stub)
    sys.modules['nam.data'].wav_to_tensor = None
    from nam.models._from_nam import init_from_nam
    return init_from_nam


_MODEL_CACHE: Dict[str, Any] = {}


def _load_model(effect):
    key = effect['name'] + ':' + str(id(effect['data']))
    if key not in _MODEL_CACHE:
        init_from_nam = import_nam_models()
        model = init_from_nam(effect['data'])
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def render_nam(clean, sample_rate: int, effect, gain_db: float = 0.0) -> np.ndarray:
    """Run audio through a real NAM capture with the `nam` package. Output is at `sample_rate`.
    `gain_db` is applied to the input first (how hard the guitar hits the amp)."""
    try:
        import torch
    except ImportError as exc:
        raise EffectError(f"{effect['name']}: PyTorch is not installed ({exc})")
    try:
        model = _load_model(effect)
        model_rate = float(getattr(model, 'sample_rate', None) or effect.get('sample_rate') or 48000.0)
        signal = resample(clean, sample_rate, model_rate) * (10.0 ** (gain_db / 20.0))
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = model(x, pad_start=True)
        out = y.detach().cpu().numpy().reshape(-1)
    except ImportError as exc:
        raise EffectError(f"{effect['name']}: the 'neural-amp-modeler' package cannot be loaded ({exc})")
    except Exception as exc:
        raise EffectError(f"{effect['name']}: failed to run capture ({exc})")
    return resample(out, model_rate, sample_rate)


def render_effect(clean, sample_rate: int, effect, gain_db: float = 0.0) -> np.ndarray:
    if effect.get('kind') == 'nam':
        return render_nam(clean, sample_rate, effect, gain_db)
    return render_fixture(clean, effect)
