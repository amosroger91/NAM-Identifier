import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.effects import EffectError, list_captures, load_effect, resample
from nam_tone_match.match import make_clean_di, rank_captures, render_capture


class TestEffects(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = os.path.join(self.tmp.name, 'models')
        self.data = os.path.join(self.tmp.name, 'data')
        os.makedirs(self.models)

    def put(self, name, obj):
        with open(os.path.join(self.models, name), 'w') as handle:
            json.dump(obj, handle)

    def test_missing_weights_raises_clear_error(self):
        self.put('bad.nam', {'version': '0.5.0', 'architecture': 'WaveNet', 'config': {}})
        with self.assertRaises(EffectError) as ctx:
            load_effect(os.path.join(self.models, 'bad.nam'))
        self.assertIn("'weights'", str(ctx.exception))
        self.assertIn('bad.nam', str(ctx.exception))

    def test_unknown_fixture_type(self):
        self.put('x.nam', {'kind': 'test-effect', 'type': 'fuzz'})
        with self.assertRaises(EffectError):
            load_effect(os.path.join(self.models, 'x.nam'))

    def test_list_captures_only_nam_sorted(self):
        for n in ('b.nam', 'a.nam'):
            self.put(n, {'kind': 'test-effect', 'type': 'identity'})
        open(os.path.join(self.models, 'README.md'), 'w').close()
        self.assertEqual(list_captures(self.models), ['a.nam', 'b.nam'])
        self.assertEqual(list_captures(os.path.join(self.tmp.name, 'nope')), [])

    def test_rank_captures_and_render(self):
        for n, t in (('identity.nam', 'identity'), ('lowpass.nam', 'lowpass'), ('gain.nam', 'gain')):
            self.put(n, {'kind': 'test-effect', 'type': t})
        clean = make_clean_di(48000, 2.0).astype(np.float64)
        target = np.convolve(clean, np.ones(8) / 8, mode='same')
        results = rank_captures(target, 48000, self.models, self.data)
        self.assertEqual(results[0]['name'], 'lowpass.nam')
        self.assertEqual({r['name'] for r in results}, {'identity.nam', 'lowpass.nam', 'gain.nam'})
        self.assertTrue(os.path.isfile(os.path.join(self.data, 'clean-di.wav')))
        samples, rate = render_capture('gain.nam', self.models, self.data)
        self.assertEqual(rate, 48000)
        with self.assertRaises(EffectError):
            render_capture('../evil.nam', self.models, self.data)
        with self.assertRaises(EffectError):
            render_capture('absent.nam', self.models, self.data)

    def test_resample_length(self):
        self.assertEqual(len(resample(np.zeros(44100), 44100, 48000)), 48000)


if __name__ == '__main__':
    unittest.main()
