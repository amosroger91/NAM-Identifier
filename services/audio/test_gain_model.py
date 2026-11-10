import unittest

import numpy as np

from nam_tone_match import gain_model
from nam_tone_match.phrases import make_phrase
from nam_tone_match.tone import classify_dynamics

RATE = gain_model.RATE


class TestGainModel(unittest.TestCase):
    def setUp(self):
        self.phrase = make_phrase(5, 6.0).astype(np.float64)

    def test_features_shape_and_silence(self):
        vec = gain_model.features(self.phrase)
        self.assertEqual(vec.shape, (len(gain_model.FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertIsNone(gain_model.features(np.zeros(RATE)))
        self.assertIsNone(gain_model.features(self.phrase[:1000]))

    def test_features_ignore_level(self):
        np.testing.assert_allclose(gain_model.features(self.phrase), gain_model.features(0.1 * self.phrase), atol=1e-6)

    def test_calibration_matches_feature_list(self):
        cal = gain_model.calibration()
        self.assertIsNotNone(cal)
        self.assertEqual(cal['features'], list(gain_model.FEATURE_NAMES))
        for key in ('mean', 'scale', 'coefficients'):
            self.assertEqual(len(cal[key]), len(gain_model.FEATURE_NAMES))

    def test_clipping_reads_dirtier_than_clean(self):
        clean = gain_model.predict_thd(self.phrase)
        driven = gain_model.predict_thd(np.tanh(12.0 * self.phrase))
        self.assertIsNotNone(clean)
        self.assertGreater(driven, clean * 2)

    def test_classify_dynamics_uses_the_phrase_model_on_the_harmonic_scale(self):
        gain = classify_dynamics(np.tanh(12.0 * self.phrase))
        self.assertEqual(gain['method'], 'phrase-model')
        self.assertAlmostEqual(gain['score'], min(1.0, gain_model.predict_thd(np.tanh(12.0 * self.phrase)) / 0.55), places=3)
        self.assertIn(gain['label'], ('crunch', 'overdrive', 'high_gain'))


if __name__ == '__main__':
    unittest.main()
