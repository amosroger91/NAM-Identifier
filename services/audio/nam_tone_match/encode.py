# Write audio to disk

import wave

import numpy as np


def write_wav(path: str, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Write 16-bit PCM WAV from mono samples or a frame-by-channel array."""
    audio = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype('<i2')
    if audio.ndim == 1 and channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    elif audio.ndim > 1 and (audio.ndim != 2 or audio.shape[1] != channels):
        raise ValueError('Audio channel count does not match the sample array')
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
