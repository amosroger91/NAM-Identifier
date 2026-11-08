import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.effects import EffectError, resample, unwrap_slimmable
from nam_tone_match.match import make_probe_di, tone_distance, tone_profile

RATE = 48000


class TestTone(unittest.TestCase):
    def test_probe_is_deterministic_and_bounded(self):
        a = make_probe_di(RATE, 3.0)
        b = make_probe_di(RATE, 3.0)
        self.assertEqual(len(a), 3 * RATE)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(float(np.max(np.abs(a))), 0.61)
        self.assertGreater(float(np.sqrt(np.mean(a ** 2))), 0.01)

    def test_tone_distance_ignores_level_but_sees_filtering(self):
        probe = make_probe_di(RATE, 4.0).astype(np.float64)
        same = tone_distance(tone_profile(probe, RATE), tone_profile(probe * 0.1, RATE))
        dull = np.convolve(probe, np.ones(16) / 16, mode='same')
        different = tone_distance(tone_profile(probe, RATE), tone_profile(dull, RATE))
        self.assertLess(same, 0.01)
        self.assertGreater(different, 2.0)

    def test_tone_profile_survives_silence(self):
        profile = tone_profile(np.zeros(RATE), RATE)
        self.assertTrue(np.all(np.isfinite(profile)))

    def test_resample_preserves_tone(self):
        t = np.arange(44100) / 44100
        up = resample(np.sin(2 * np.pi * 1000 * t), 44100, 48000)
        self.assertEqual(len(up), 48000)
        spectrum = np.abs(np.fft.rfft(up))
        self.assertAlmostEqual(float(np.argmax(spectrum)) * RATE / len(up), 1000.0, delta=2.0)
        self.assertAlmostEqual(float(np.max(np.abs(up))), 1.0, delta=0.02)

    def test_slimmable_container_uses_largest_submodel(self):
        small = {'architecture': 'WaveNet', 'config': {}, 'weights': [1], 'version': '0.7.0'}
        big = {'architecture': 'WaveNet', 'config': {}, 'weights': [1, 2], 'version': '0.7.0'}
        outer = {'architecture': 'SlimmableContainer', 'sample_rate': 48000,
                 'config': {'submodels': [{'max_value': 1.0, 'model': big}, {'max_value': 0.5, 'model': small}]}}
        picked = unwrap_slimmable(outer)
        self.assertEqual(picked['weights'], [1, 2])
        self.assertEqual(picked['sample_rate'], 48000)
        with self.assertRaises(EffectError):
            unwrap_slimmable({'architecture': 'SlimmableContainer', 'config': {'submodels': []}})


if __name__ == '__main__':
    unittest.main()
