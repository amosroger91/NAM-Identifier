from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import tone_benchmark as TB


def row(truth, pred, rank=1, top1='a', seen=False, condition='dry', adjectives=(), card=()):
    return {'truth': truth, 'pred': pred, 'rank': rank, 'top1': top1, 'seen': seen, 'condition': condition,
            'method': 'dynamics', 'adjectives': list(adjectives), 'truthAdjectives': list(card),
            'clapAdjectives': [], 'brightnessDb': 0.0, 'truthBrightnessDb': 0.0}


class TestToneBenchmarkMetrics(unittest.TestCase):
    def test_gain_stats_counts_distance_bias_and_uncertain(self):
        stats = TB.gain_stats([row('clean', 'clean'), row('clean', 'crunch'), row('crunch', 'clean'),
                               row('edge', 'uncertain')])
        self.assertEqual(stats['n'], 4)
        self.assertEqual(stats['uncertain'], 1)
        self.assertAlmostEqual(stats['exact'], 1 / 3, places=3)
        self.assertAlmostEqual(stats['meanDistance'], 4 / 3, places=3)
        self.assertAlmostEqual(stats['bias'], 0.0, places=3)

    def test_search_stats_reports_retrieval_and_winner_concentration(self):
        stats = TB.search_stats([row('clean', 'clean', rank=1, top1='x'), row('clean', 'clean', rank=4, top1='x'),
                                 row('clean', 'clean', rank=10, top1='y')])
        self.assertAlmostEqual(stats['top1'], 1 / 3, places=3)
        self.assertAlmostEqual(stats['top5'], 2 / 3, places=3)
        self.assertAlmostEqual(stats['mrr'], (1 + 0.25 + 0.1) / 3, places=3)
        self.assertEqual(stats['mostCommonWinner'], 'x')
        self.assertEqual(stats['distinctWinners'], 2)

    def test_adjective_agreement_compares_query_to_card(self):
        stats = TB.adjective_stats([row('clean', 'clean', adjectives=['dark'], card=[]),
                                    row('clean', 'clean', adjectives=['dark'], card=['dark'])])
        self.assertEqual(stats['words']['dark']['rate'], 1.0)
        self.assertEqual(stats['words']['dark']['truthRate'], 0.5)
        self.assertEqual(stats['words']['dark']['agreement'], 0.5)

    def test_plate_keeps_length_and_level(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(TB.RATE) * 0.1).astype(np.float32)
        y = TB.plate(x, 0.4, 1.5, 3)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertAlmostEqual(float(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))), 1.0, delta=0.35)


if __name__ == '__main__':
    unittest.main()
