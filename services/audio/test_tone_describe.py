import json
import os
import sys
import tempfile
import unittest

import numpy as np

os.environ['NAM_CLAP'] = '0'
sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.clap_embed import NullEmbedder, get_embedder
from nam_tone_match.cli import handle
from nam_tone_match.encode import write_wav
from nam_tone_match.tone import compare_samples, describe_samples, similarity_percent, tone_snapshot
from nam_tone_match.tone_search import _canonical, search_captures

RATE = 48000
NULL = NullEmbedder('off')


def _sine(gain=1.0, freq=220.0):
    time = np.arange(RATE) / RATE
    return (0.5 * np.sin(2.0 * np.pi * freq * time) * gain).astype(np.float64)


class TestDescribe(unittest.TestCase):
    def test_harmonic_ladder(self):
        sine = _sine()
        edge = np.tanh(1.5 * sine)
        crunch = np.tanh(3.0 * sine)
        overdrive = np.tanh(8.0 * sine)
        high_gain = 0.4 * np.sign(np.sin(2.0 * np.pi * 110.0 * np.arange(RATE) / RATE))
        fuzz = np.clip(sine * 12.0, -0.15, 0.15)
        labels = [
            (sine, 'clean'),
            (edge, 'edge'),
            (crunch, 'crunch'),
            (overdrive, 'overdrive'),
            (high_gain, 'high_gain'),
            (fuzz, 'fuzz'),
        ]
        scores = []
        for audio, label in labels:
            report = describe_samples(audio, RATE, embedder=NULL)
            self.assertEqual(report['gain']['label'], label, label)
            self.assertEqual(report['gain']['method'], 'harmonics')
            scores.append(report['gain']['score'])
        self.assertEqual(scores, sorted(scores))

    def test_brightness_compression_and_silence(self):
        generator = np.random.default_rng(0)
        noise = generator.normal(0.0, 0.1, RATE)
        bright = noise - np.convolve(noise, np.ones(50) / 50.0, mode='same')
        dark = np.convolve(noise, np.ones(200) / 200.0, mode='same')
        bright_report = describe_samples(bright, RATE, embedder=NULL)
        dark_report = describe_samples(dark, RATE, embedder=NULL)
        self.assertIn('bright', bright_report['adjectives'])
        self.assertIn('dark', dark_report['adjectives'])
        self.assertGreater(bright_report['features']['brightnessDb'], dark_report['features']['brightnessDb'] + 20)
        self.assertEqual(bright_report['gain']['label'], 'uncertain')

        pluck = _sine() * np.exp(-4.0 * np.arange(RATE) / RATE)
        limited = np.clip(pluck * 10.0, -0.2, 0.2)
        limited_report = describe_samples(limited, RATE, embedder=NULL)
        self.assertIn(limited_report['gain']['label'], ('high_gain', 'fuzz'))
        self.assertIn('compressed', limited_report['adjectives'])
        self.assertIn('saturated', limited_report['adjectives'])
        self.assertGreater(
            describe_samples(pluck, RATE, embedder=NULL)['features']['crestDb'],
            limited_report['features']['crestDb'] + 5,
        )

        silent = describe_samples(np.zeros(RATE), RATE, embedder=NULL)
        self.assertEqual(silent['gain']['label'], 'uncertain')
        self.assertEqual(silent['adjectives'], [])

    def test_similarity_is_100_for_the_same_audio_and_lower_across_gain(self):
        sine = describe_samples(_sine(), RATE, embedder=NULL)
        crunch = describe_samples(np.tanh(3.0 * _sine()), RATE, embedder=NULL)
        fuzz = describe_samples(np.clip(_sine() * 12.0, -0.15, 0.15), RATE, embedder=NULL)
        self.assertEqual(similarity_percent(sine, sine), 100)
        self.assertLess(similarity_percent(sine, fuzz), 50)
        self.assertLess(similarity_percent(sine, fuzz), similarity_percent(sine, crunch))
        compared = compare_samples(_sine(), RATE, _sine(), RATE, embedder=NULL)
        self.assertEqual(compared['similarity'], 100)
        self.assertIsNone(compared['clapCosine'])
        self.assertFalse(compared['left']['clap']['available'])

    def test_clap_can_name_an_uncertain_clip_and_not_a_measured_one(self):
        class Fake:
            available = True
            model_id = 'fake-clap'
            reason = None

            def embed_audio(self, samples, rate):
                rms = float(np.sqrt(np.mean(np.asarray(samples) ** 2)) or 0.0)
                return np.array([rms, 1.0 - rms])

            def score_prompts(self, embedding):
                return [
                    {'label': 'fuzz', 'group': 'gain', 'text': 'fuzz guitar', 'score': 0.42},
                    {'label': 'clean', 'group': 'gain', 'text': 'clean electric guitar', 'score': 0.10},
                    {'label': 'bright', 'group': 'adjective', 'text': 'bright electric guitar', 'score': 0.31},
                ]

        generator = np.random.default_rng(1)
        noise = generator.normal(0.0, 0.2, RATE)
        noise = noise - np.convolve(noise, np.ones(40) / 40.0, mode='same')
        named = describe_samples(noise, RATE, embedder=Fake())
        self.assertEqual(named['gain']['method'], 'clap')
        self.assertEqual(named['gain']['label'], 'fuzz')
        self.assertTrue(named['clap']['available'])
        held = describe_samples(_sine(), RATE, embedder=Fake())
        self.assertEqual(held['gain']['label'], 'clean')
        self.assertEqual(held['gain']['method'], 'harmonics')

    def test_search_ranks_the_capture_that_matches_the_reference(self):
        phrase, _sine_probe = _canonical()
        dulled = np.convolve(phrase, np.ones(8) / 8.0, mode='same')
        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, 'models')
            data = os.path.join(root, 'data')
            os.makedirs(models)
            for kind in ('identity', 'lowpass', 'gain'):
                with open(os.path.join(models, kind + '.nam'), 'w', encoding='utf-8') as fh:
                    json.dump({'kind': 'test-effect', 'type': kind}, fh)
            with open(os.path.join(models, 'broken.nam'), 'w', encoding='utf-8') as fh:
                fh.write('not json')
            found = search_captures(dulled, RATE, models, data, embedder=NULL)
            self.assertEqual(found['results'][0]['name'], 'lowpass.nam')
            # Cards hold the measured probe gain; a query's gain is the phrase model's estimate of it,
            # so even the card's own phrase scores a little under 100.
            self.assertGreaterEqual(found['results'][0]['similarity'], 90)
            by_name = {row['name']: row for row in found['results']}
            self.assertGreater(by_name['lowpass.nam']['similarity'], by_name['identity.nam']['similarity'])
            self.assertEqual(by_name['lowpass.nam']['gain']['label'], 'clean')
            self.assertEqual(found['skipped'][0]['name'], 'broken.nam')
            self.assertEqual(found['built'], 3)
            query = tone_snapshot(describe_samples(dulled, RATE, embedder=NULL))
            reused = handle({'command': 'tone-search', 'guitar': os.path.join(root, 'unavailable.wav'),
                             'modelsDir': models, 'dataDir': data, 'toneQuery': query})
            self.assertEqual(reused['results'], found['results'])
            self.assertEqual(reused['query'], found['query'])
            self.assertNotIn('vector', reused['query'])
            top = reused['results'][0]
            self.assertLess(abs(sum(top['scoreBreakdown'].values()) - top['similarity']), 0.7)
            again = search_captures(phrase, RATE, models, data, embedder=NULL)
            self.assertEqual(again['cached'], 3)
            self.assertEqual(again['built'], 0)
            clean = {row['name']: row['similarity'] for row in again['results']}
            self.assertGreater(clean['identity.nam'], clean['lowpass.nam'])
            self.assertEqual(clean['identity.nam'], clean['gain.nam'])

    def test_cli_describe_and_compare(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, 'a.wav')
            second = os.path.join(root, 'b.wav')
            write_wav(first, _sine(), RATE)
            write_wav(second, np.clip(_sine() * 12.0, -0.15, 0.15), RATE)
            described = handle({'command': 'describe', 'path': first})
            self.assertEqual(described['gain']['label'], 'clean')
            self.assertFalse(described['clap']['available'])
            compared = handle({'command': 'compare', 'a': first, 'b': first})
            self.assertEqual(compared['similarity'], 100)
            apart = handle({'command': 'compare', 'a': first, 'b': second})
            self.assertLess(apart['similarity'], 50)
            with self.assertRaises(ValueError):
                handle({'command': 'describe'})

    def test_clap_stays_off_unless_requested(self):
        os.environ['NAM_CLAP'] = '0'
        embedder = get_embedder()
        self.assertFalse(embedder.available)
        self.assertIn('NAM_CLAP', embedder.reason)


if __name__ == '__main__':
    unittest.main()
