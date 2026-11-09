import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import measure_accuracy as measure
from nam_tone_match.match import tone_profile


class TestControlledBenchmark(unittest.TestCase):
    def test_baseline_scores_match_historical_spectral_distance(self):
        time = np.arange(8192, dtype=np.float32) / 48000
        signals = [np.sin(2 * np.pi * hz * time) for hz in (220, 440, 330)]
        names = ['a', 'b']
        features = [measure.B.features(signal) for signal in signals]
        bank = {name: {0.0: [feat]} for name, feat in zip(names, features)}
        score = measure.scoring.make_scorer(bank, names)
        reference = tone_profile(signals[-1], 48000)
        expected = [np.sqrt(np.mean((reference - tone_profile(signal, 48000)) ** 2))
                    for signal in signals[:2]]
        np.testing.assert_allclose(score(features[-1]), expected, atol=1e-5)

    def test_multi_window_requires_one_trim_and_ties_use_capture_order(self):
        def feat(value):
            return {'b': np.array([value]), 'dyn': np.zeros(4)}

        names = ['first', 'second']
        bank = {name: {0: [feat(0)], 6: [feat(10)]} for name in names}
        score = measure.scoring.make_scorer(bank, names, key='b', windows=True)
        query = {**feat(5), 'win': np.array([[0], [10]])}
        ranks, scores, winners = measure.evaluate(score, [{'truth': 'second', 'feats': query}], names)
        np.testing.assert_array_equal(scores, [[5, 5]])
        np.testing.assert_array_equal(ranks, [2])
        np.testing.assert_array_equal(winners, [0])

    def test_cache_identity_changes_with_input_renderer_gain_and_normalization(self):
        baseline = ['capture', 'input', 0.0, {'version': 1}, 48000, 0.5]
        original = measure.cache_path(*baseline)
        for index, value in enumerate(('other', 'different', 6, {'version': 2}, 44100, 0.6)):
            changed = baseline.copy()
            changed[index] = value
            self.assertNotEqual(original, measure.cache_path(*changed))

    def test_dev_build_omits_black_audio(self):
        phrases = {name: {'name': name, 'instrument': instrument, 'role': role}
                   for name, instrument, role in [('bank', 'green', 'bank'),
                                                   ('dev', 'green', 'query'), ('held', 'black', 'query')]}
        with patch.object(measure.B, 'bank_inputs', return_value={}), \
                patch.object(measure.B, 'query_inputs', return_value={}), \
                patch.object(measure, 'materialize', return_value=np.ones(2)) as materialize:
            _, queries = measure.prepare({'phrases': phrases, 'audio': {}}, Path('.'), ('green', 'synth'))
        self.assertEqual(set(queries), {'green', 'synth'})
        self.assertEqual([call.args[0]['name'] for call in materialize.call_args_list], ['bank', 'dev'])

    def test_final_rejects_tampered_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'selection.json'
            payload = {'identity': {'protocol_id': 'frozen'}, 'chosen': 'method', 'gate': measure.GATE}
            measure.save_json(path, {**payload, 'selection_sha256': measure.digest(payload)})
            measure.read_selection(path, payload['identity'])
            tampered = json.loads(path.read_text())
            tampered['chosen'] = 'alternative'
            measure.save_json(path, tampered)
            with self.assertRaisesRegex(ValueError, 'Frozen selection differs'):
                measure.read_selection(path, payload['identity'])
