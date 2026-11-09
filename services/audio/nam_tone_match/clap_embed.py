"""Optional LAION CLAP embeddings.

Off unless ``NAM_CLAP=1`` (or ``NAM_CLAP_MODEL`` points at a local model).
The default model id is ``laion/clap-htsat-unfused``. Loading it downloads the
weights on first use and imports ``transformers``. A missing package or a failed
download leaves descriptions on the DSP measurements.
"""

import os

import numpy as np

from .effects import resample
from .tone import ANALYSIS_RATE

PROMPTS = (
    ('clean', 'gain', 'clean electric guitar'),
    ('edge', 'gain', 'edge of breakup electric guitar'),
    ('crunch', 'gain', 'crunchy electric guitar'),
    ('overdrive', 'gain', 'overdriven electric guitar'),
    ('high_gain', 'gain', 'high gain heavily distorted electric guitar'),
    ('fuzz', 'gain', 'fuzz guitar'),
    ('bright', 'adjective', 'bright electric guitar'),
    ('dark', 'adjective', 'dark electric guitar'),
    ('mid-heavy', 'adjective', 'mid-heavy electric guitar'),
    ('compressed', 'adjective', 'compressed electric guitar'),
    ('glassy', 'adjective', 'glassy clean electric guitar'),
    ('saturated', 'adjective', 'saturated distorted electric guitar'),
)

_CACHE = {}


class NullEmbedder:
    available = False

    def __init__(self, reason):
        self.reason = reason
        self.model_id = None

    def embed_audio(self, samples, rate):
        return None

    def score_prompts(self, embedding):
        return []


class ClapEmbedder:
    available = True

    def __init__(self, model, processor, model_id):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.reason = None
        self._text = None

    @classmethod
    def load(cls):
        import torch
        from transformers import ClapModel, ClapProcessor

        model_id = os.environ.get('NAM_CLAP_MODEL') or 'laion/clap-htsat-unfused'
        processor = ClapProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
        model.eval()
        return cls(model, processor, model_id)

    def embed_audio(self, samples, rate):
        import torch

        audio = np.nan_to_num(np.asarray(samples, dtype=np.float32).reshape(-1))
        if audio.size < 2:
            return None
        if int(rate) != 48000:
            audio = resample(audio, float(rate), 48000).astype(np.float32)
        # CLAP was trained on clips up to about 10 seconds.
        if audio.size > 48000 * 7:
            audio = audio[:48000 * 7]
        inputs = self.processor(audios=audio, sampling_rate=48000, return_tensors='pt')
        with torch.no_grad():
            features = self.model.get_audio_features(**inputs)
        vector = features[0].detach().cpu().numpy().astype(np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def _text_matrix(self):
        import torch

        if self._text is None:
            texts = [item[2] for item in PROMPTS]
            inputs = self.processor(text=texts, return_tensors='pt', padding=True)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            matrix = features.detach().cpu().numpy().astype(np.float64)
            matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            self._text = matrix
        return self._text

    def score_prompts(self, embedding):
        if embedding is None:
            return []
        vector = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return []
        scores = self._text_matrix() @ (vector / norm)
        ranked = []
        for score, (label, group, text) in sorted(zip(scores.tolist(), PROMPTS), key=lambda item: -item[0]):
            ranked.append({'label': label, 'group': group, 'text': text, 'score': round(float(score), 4)})
        return ranked


def get_embedder():
    """Return a cached CLAP embedder, or a null embedder that explains why."""
    from .tone import clap_requested

    key = (os.environ.get('NAM_CLAP', ''), os.environ.get('NAM_CLAP_MODEL', ''))
    if _CACHE.get('key') == key and 'embedder' in _CACHE:
        return _CACHE['embedder']
    _CACHE.clear()
    _CACHE['key'] = key
    if not clap_requested():
        embedder = NullEmbedder(
            'CLAP is off. Set NAM_CLAP=1 to add LAION CLAP embeddings. '
            'Requires: python -m pip install transformers')
    else:
        try:
            embedder = ClapEmbedder.load()
        except Exception as exc:
            embedder = NullEmbedder(f'CLAP unavailable: {exc}')
    _CACHE['embedder'] = embedder
    return embedder
