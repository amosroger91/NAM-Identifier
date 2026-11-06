import unittest
import numpy as np
import os
import sys

# Add the nam_tone_match package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nam_tone_match'))

from slice import slice_audio

class TestSlice(unittest.TestCase):
    def test_slice_audio_normal(self):
        # Create a simple audio array
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Test normal slicing
        sliced_audio = slice_audio(audio, start_sec=0.5, end_sec=1.5, sample_rate=sample_rate)
        
        # Should be 1 second of audio (44100 samples)
        expected_length = int(sample_rate * 1.0)  # 1 second
        self.assertEqual(len(sliced_audio), expected_length)
        
    def test_slice_audio_inverted_range(self):
        # Create a simple audio array
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Test inverted range - should raise ValueError
        with self.assertRaises(ValueError) as context:
            slice_audio(audio, start_sec=1.5, end_sec=0.5, sample_rate=sample_rate)
        
        self.assertIn("Inverted range", str(context.exception))
        
    def test_slice_audio_past_end(self):
        # Create a simple audio array
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Test range past end - should raise ValueError
        with self.assertRaises(ValueError) as context:
            slice_audio(audio, start_sec=0.5, end_sec=3.0, sample_rate=sample_rate)
        
        self.assertIn("Range extends past end of audio", str(context.exception))

if __name__ == '__main__':
    unittest.main()
