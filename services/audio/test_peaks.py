import unittest
import numpy as np
import os
import sys

# Add the nam_tone_match package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nam_tone_match'))

from peaks import compute_peaks

class TestPeaks(unittest.TestCase):
    def test_compute_peaks(self):
        # Create a simple audio array with known peaks
        sample_rate = 44100
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a signal with clear peaks
        audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        
        peaks = compute_peaks(audio, sample_rate)
        
        # Should return about 1000 peaks
        self.assertGreater(len(peaks), 900)  # At least 900 peaks
        self.assertLess(len(peaks), 1100)   # At most 1100 peaks
        
        # All peaks should be in range [0, 1]
        self.assertTrue(np.all(peaks >= 0.0) and np.all(peaks <= 1.0))
        
    def test_compute_peaks_empty_audio(self):
        # Test with empty audio
        audio = np.array([])
        peaks = compute_peaks(audio, 44100)
        self.assertEqual(len(peaks), 0)
        
    def test_compute_peaks_exact_buckets(self):
        # Test with audio that has exact bucket sizes
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        peaks = compute_peaks(audio, sample_rate, bucket_count=1000)
        
        # Should return exactly 1000 peaks
        self.assertEqual(len(peaks), 1000)
        
        # All peaks should be in range [0, 1]
        self.assertTrue(np.all(peaks >= 0.0) and np.all(peaks <= 1.0))

if __name__ == '__main__':
    unittest.main()
