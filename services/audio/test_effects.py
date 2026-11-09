import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.audition import RATE, SECONDS, prepare_source, render_audition
from nam_tone_match.decode import decode_wav
from nam_tone_match.effects import EffectError, list_captures, load_effect, resample
from nam_tone_match.encode import write_wav
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

    def test_audition_pairs_the_bundled_recording_with_the_effect(self):
        self.put('lowpass.nam', {'kind': 'test-effect', 'type': 'lowpass'})
        with patch.dict(os.environ, {'NAM_AUDITION_DI': ''}):
            source = prepare_source(self.tmp.name)
            self.assertEqual(prepare_source(self.tmp.name), source)
        audio, rate = render_audition(os.path.join(self.models, 'lowpass.nam'), source, compare=True)
        raw, _ = decode_wav(source)
        self.assertEqual(rate, RATE)
        self.assertEqual(audio.shape, (SECONDS * rate, 2))
        wet, dry = audio.T
        np.testing.assert_allclose(dry, raw, atol=1e-7)
        self.assertFalse(np.allclose(wet, dry))
        self.assertAlmostEqual(float(np.mean(wet ** 2)), float(np.mean(dry ** 2)), places=7)

    def test_audition_accepts_a_performed_di(self):
        take = os.path.join(self.tmp.name, 'take.wav')
        write_wav(take, np.full(24000, 0.4), 24000)
        self.put('gain.nam', {'kind': 'test-effect', 'type': 'gain'})
        with patch.dict(os.environ, {'NAM_AUDITION_DI': take}):
            source = prepare_source(self.tmp.name)
        audio, rate = render_audition(os.path.join(self.models, 'gain.nam'), source)
        self.assertEqual(rate, RATE)
        self.assertEqual(len(audio), rate)
        self.assertAlmostEqual(audio[rate // 2], 0.125, places=4)


class TestResampleParity(unittest.TestCase):
    def test_upsampling_preserves_cosines_across_even_and_odd_lengths(self):
        # An even input's Nyquist cosine becomes an ordinary positive/negative
        # frequency pair when upsampled, even when the rFFT bin count stays equal.
        for n, m, frequency in ((4, 5, 2), (6, 7, 3), (8, 9, 4), (8, 12, 4), (9, 10, 4)):
            with self.subTest(n=n, m=m):
                source = np.cos(2 * np.pi * frequency * np.arange(n) / n)
                expected = np.cos(2 * np.pi * frequency * np.arange(m) / m)
                np.testing.assert_allclose(resample(source, n, m), expected, atol=1e-12)

    def test_downsampling_merges_nyquist_and_removes_out_of_band_cosines(self):
        for n, m, frequency in ((9, 8, 4), (16, 8, 4), (9, 7, 3), (8, 7, 3)):
            with self.subTest(n=n, m=m):
                source = np.cos(2 * np.pi * frequency * np.arange(n) / n)
                expected = np.cos(2 * np.pi * frequency * np.arange(m) / m)
                np.testing.assert_allclose(resample(source, n, m), expected, atol=1e-12)
        np.testing.assert_allclose(resample((-1.0) ** np.arange(8), 8, 7), np.zeros(7), atol=1e-12)


class TestRankSkipsBadCaptures(unittest.TestCase):
    def test_bad_capture_is_reported_not_fatal(self):
        import json, tempfile
        from nam_tone_match.match import rank_captures
        with tempfile.TemporaryDirectory() as tmp:
            models, data = os.path.join(tmp, 'models'), os.path.join(tmp, 'data')
            os.makedirs(models)
            json.dump({'kind': 'test-effect', 'type': 'identity'}, open(os.path.join(models, 'ok.nam'), 'w'))
            json.dump({'kind': 'test-effect', 'type': 'lowpass'}, open(os.path.join(models, 'lp.nam'), 'w'))
            json.dump({'version': '0.5.0', 'architecture': 'WaveNet', 'config': {}},
                      open(os.path.join(models, 'broken.nam'), 'w'))      # no weights
            report = {}
            ranked = rank_captures(make_clean_di(48000, 3.0), 48000, models, data, report)
            self.assertEqual(sorted(r['name'] for r in ranked), ['lp.nam', 'ok.nam'])
            self.assertEqual(report['searched'], 3)
            self.assertEqual(report['ranked'], 2)
            self.assertEqual([k['name'] for k in report['skipped']], ['broken.nam'])
            self.assertIn('weights', report['skipped'][0]['error'])
            self.assertIn('probeInputTrimDb', ranked[0])


if __name__ == '__main__':
    unittest.main()
