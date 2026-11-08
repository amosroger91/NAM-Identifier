import os
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nam_tone_match.decode import decode_file, decode_mp3, decode_wav
from nam_tone_match.encode import write_wav


def write_raw_wav(path, pcm, channels, width, rate):
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm)


class TestDecode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_wav_roundtrip_within_1e4(self):
        rate = 44100
        samples = (0.8 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype(np.float32)
        write_wav(self.path('a.wav'), samples, rate)
        decoded, decoded_rate = decode_wav(self.path('a.wav'))
        self.assertEqual(decoded_rate, rate)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(len(decoded), len(samples))
        self.assertLess(float(np.max(np.abs(decoded - samples))), 1e-4)

    def test_stereo_is_averaged(self):
        left = np.full(100, 8000, dtype='<i2')
        right = np.full(100, -4000, dtype='<i2')
        write_raw_wav(self.path('s.wav'), np.column_stack([left, right]).tobytes(), 2, 2, 8000)
        decoded, _ = decode_wav(self.path('s.wav'))
        self.assertEqual(len(decoded), 100)
        self.assertAlmostEqual(float(decoded[0]), 2000 / 32768, places=5)

    def test_24_bit(self):
        values = [0, 4194304, -4194304]  # 0, +0.5, -0.5
        raw = b''.join(int(v & 0xFFFFFF).to_bytes(3, 'little') for v in values)
        write_raw_wav(self.path('t.wav'), raw, 1, 3, 8000)
        decoded, _ = decode_wav(self.path('t.wav'))
        np.testing.assert_allclose(decoded, [0, 0.5, -0.5], atol=1e-6)

    def test_decode_file_sniffs_wav(self):
        write_wav(self.path('x.source'), np.zeros(10), 8000)
        decoded, rate = decode_file(self.path('x.source'))
        self.assertEqual((len(decoded), rate), (10, 8000))

    def test_mp3_decoder_exists(self):
        self.assertTrue(callable(decode_mp3))


if __name__ == '__main__':
    unittest.main()
