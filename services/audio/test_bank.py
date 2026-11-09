import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match import bank, drive
from nam_tone_match.effects import resample
from nam_tone_match.encode import write_wav
from nam_tone_match.match import make_probe_di, rank_captures


class TestBank(unittest.TestCase):
    def test_profile_is_level_invariant_and_sees_filtering(self):
        x = make_probe_di(48000, 5.0).astype(np.float64)
        x = x + 1e-3 * np.random.default_rng(0).standard_normal(len(x))   # keep every band above the log floor
        same = np.sqrt(np.mean((bank.tone_profile_db(x) - bank.tone_profile_db(x * 0.05)) ** 2))
        dull = np.convolve(x, np.ones(16) / 16, mode='same')
        diff = np.sqrt(np.mean((bank.tone_profile_db(x) - bank.tone_profile_db(dull)) ** 2))
        self.assertLess(same, 0.01)
        self.assertGreater(diff, 2.0)
        self.assertEqual(bank.tone_profile_db(x).shape, (28,))
        self.assertTrue(np.all(np.isfinite(bank.tone_profile_db(np.zeros(48000)))))

    def test_nearest_prototype(self):
        protos = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        d, trim = bank.nearest_prototype(np.array([3.0, 3.0]), protos, np.array([0.0, 6.0], dtype=np.float32))
        self.assertAlmostEqual(d, np.sqrt(0.5), places=5)
        self.assertEqual(trim, 6.0)

    def test_drive_profile_uses_the_measured_state_at_each_trim(self):
        query = np.zeros(28)
        profiles = np.stack([np.full(28, 0.5), query])
        trims = np.array([-6, 6])
        curve = np.array([0, 0, 0.5, 1, 1, 1])
        model = drive.calibration()
        self.assertEqual(drive.compare_tone(query, profiles, trims, 0.05, curve, 1, model)[1], -6.0)
        self.assertEqual(drive.compare_tone(query, profiles, trims, 0.95, curve, 1, model)[1], 6.0)
        self.assertEqual(drive.compare_tone(query, profiles, trims, None, curve, 1, model),
                         bank.nearest_prototype(query, profiles, trims))

    def test_drive_ranker_keeps_unmeasured_captures(self):
        from unittest.mock import patch
        from nam_tone_match.effects import EffectError

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'identity.nam')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump({'kind': 'test-effect', 'type': 'identity'}, handle)
            report = {}
            x = make_probe_di(48000, 1)
            with patch.object(bank, 'make_phrase_source', return_value=lambda: [x]), \
                    patch.object(drive, 'capture_drive', side_effect=EffectError('probe failed')):
                results = drive.rank_captures(x, 48000, tmp, tmp, report, [])
            self.assertEqual([r['name'] for r in results], ['identity.nam'])
            self.assertEqual(report['drive']['unmeasuredCaptures'], 1)
            self.assertIn(report['drive']['label'], ('clean', 'dirty', 'uncertain'))

    def test_bank_id_changes_with_recorded_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, none = bank.bank_spec([])
            self.assertEqual(none['recordedPhrases'], 0)
            write_wav(os.path.join(tmp, 'a.wav'), np.sin(np.arange(24000) / 20.0), 48000)
            files, with_clips = bank.bank_spec([tmp])
            self.assertEqual(len(files), 1)
            self.assertEqual(with_clips['recordedPhrases'], bank.RECORDED_PHRASES)
            self.assertNotEqual(none['bankId'], with_clips['bankId'])

    def test_index_is_cached_and_ranking_uses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, data = os.path.join(tmp, 'models'), os.path.join(tmp, 'data')
            os.makedirs(models)
            for name, kind in (('id.nam', 'identity'), ('lp.nam', 'lowpass')):
                json.dump({'kind': 'test-effect', 'type': kind}, open(os.path.join(models, name), 'w'))
            old = bank.SYNTH_PHRASES
            bank.SYNTH_PHRASES = 2                      # keep the unit test quick
            bank._SYNTH_CACHE.clear()
            try:
                first = bank.index_all(['id.nam', 'lp.nam'], models, data, [])
                self.assertEqual((first['indexed'], first['cached']), (2, 0))
                again = bank.index_all(['id.nam', 'lp.nam'], models, data, [])
                self.assertEqual((again['indexed'], again['cached']), (0, 2))
                # a dull reference should land closest to the lowpass capture
                reference = np.convolve(bank.training_phrases(1)[0].astype(np.float64), np.ones(8) / 8, mode='same')
                report = {}
                ranked = rank_captures(reference, 48000, models, data, report, [])
                self.assertEqual(ranked[0]['name'], 'lp.nam')
                self.assertEqual(report['bank']['recordedPhrases'], 0)
            finally:
                bank.SYNTH_PHRASES = old
                bank._SYNTH_CACHE.clear()


if __name__ == '__main__':
    unittest.main()
