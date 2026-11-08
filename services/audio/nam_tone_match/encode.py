# Write audio to disk

import wave

import numpy as np


def write_wav(path: str, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Write float samples (-1..1) as 16-bit PCM WAV. With channels > 1 the mono signal is duplicated."""
    audio = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype('<i2')
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
