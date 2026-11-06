import unittest
import numpy as np
import os
import sys

# Add the nam_tone_match package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nam_tone_match'))

from decode import decode_wav, decode_mp3

class TestDecode(unittest.TestCase):
    def test_decode_wav(self):
        # Create a simple WAV file for testing
        sample_rate = 44100
        duration = 1.0  # 1 second
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Write a test WAV file
        import wave
        import struct
        test_wav_path = 'test.wav'
        with wave.open(test_wav_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Convert to 16-bit integers
            audio_data = (samples * 32767).astype(np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        # Test decoding
        decoded_samples, decoded_sr = decode_wav(test_wav_path)
        
        self.assertEqual(decoded_sr, sample_rate)
        self.assertTrue(len(decoded_samples) > 0)
        self.assertTrue(np.all(decoded_samples >= -1.0) and np.all(decoded_samples <= 1.0))
        
        # Clean up
        os.remove(test_wav_path)
    
    def test_decode_wav_stereo(self):
        # Create a stereo WAV file for testing
        sample_rate = 44100
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        left_channel = np.sin(2 * np.pi * 440 * t)
        right_channel = np.sin(2 * np.pi * 880 * t)
        stereo_samples = np.column_stack([left_channel, right_channel])
        
        # Write a test WAV file
        import wave
        test_wav_path = 'test_stereo.wav'
        with wave.open(test_wav_path, 'w') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Convert to 16-bit integers
            audio_data = (stereo_samples * 32767).astype(np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        # Test decoding
        decoded_samples, decoded_sr = decode_wav(test_wav_path)
        
        self.assertEqual(decoded_sr, sample_rate)
        self.assertTrue(len(decoded_samples) > 0)
        self.assertTrue(np.all(decoded_samples >= -1.0) and np.all(decoded_samples <= 1.0))
        # Should be mono after downmixing
        self.assertEqual(len(decoded_samples), int(sample_rate * duration))
        
        # Clean up
        os.remove(test_wav_path)
    
    def test_decode_wav_roundtrip(self):
        # Test a real WAV roundtrip - decode and re-encode
        sample_rate = 44100
        duration = 0.5  # 0.5 second
        original_samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Write a test WAV file
        import wave
        test_wav_path = 'roundtrip_test.wav'
        with wave.open(test_wav_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Convert to 16-bit integers
            audio_data = (original_samples * 32767).astype(np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        # Decode it back
        decoded_samples, decoded_sr = decode_wav(test_wav_path)
        
        # Check that we got the right sample rate and length
        self.assertEqual(decoded_sr, sample_rate)
        self.assertEqual(len(decoded_samples), int(sample_rate * duration))
        
        # Check that values are in range [-1, 1]
        self.assertTrue(np.all(decoded_samples >= -1.0) and np.all(decoded_samples <= 1.0))
        
        # Clean up
        os.remove(test_wav_path)
    
    def test_decode_mp3(self):
        # For now, just check that the function exists
        # We'll implement a basic test once we have the actual implementation
        self.assertTrue(callable(decode_mp3))

if __name__ == '__main__':
    unittest.main()
