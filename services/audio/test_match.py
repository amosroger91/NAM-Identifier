import unittest
import numpy as np
import os
import sys

# Add the nam_tone_match package to path
sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.match import rank_effects, make_clean_di
from nam_tone_match.effects import apply_lowpass_filter
from nam_tone_match.peaks import compute_peaks


class TestMatch(unittest.TestCase):
    def test_rank_effects(self):
        # Create a clean DI signal
        sample_rate = 44100
        duration = 1.0
        clean_di = np.array(make_clean_di(sample_rate, duration))
        
        # Create effects list
        effects = [
            {'type': 'identity'},
            {'type': 'lowpass'},
            {'type': 'gain'}
        ]
        
        # Test case 1: Target is lowpassed clean DI - should rank lowpass first
        target = apply_lowpass_filter(clean_di, sample_rate)
        
        ranked = rank_effects(target, clean_di, effects, sample_rate)
        
        # Check that lowpass effect ranks first
        self.assertEqual(ranked[0]['type'], 'lowpass')
        
        # Test case 2: Target is unchanged clean DI - should rank identity first
        target = clean_di.copy()
        ranked = rank_effects(target, clean_di, effects, sample_rate)
        
        # Check that identity effect ranks first
        self.assertEqual(ranked[0]['type'], 'identity')
        
        # Test case 3: Target is 0.5 * clean DI - should rank gain first
        target = clean_di * 0.5
        ranked = rank_effects(target, clean_di, effects, sample_rate)
        
        # Check that gain effect ranks first
        self.assertEqual(ranked[0]['type'], 'gain')

if __name__ == '__main__':
    unittest.main()
