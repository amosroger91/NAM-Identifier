# Peak extraction for waveform drawing

import numpy as np


def compute_peaks(audio: np.ndarray, sample_rate: int, bucket_count: int = 1000) -> np.ndarray:
    """Max absolute value per bucket, exactly `bucket_count` buckets, values in 0..1."""
    audio = np.asarray(audio)
    if audio.size == 0:
        return np.array([])
    magnitude = np.minimum(np.abs(audio.astype(np.float64)), 1.0)
    edges = np.linspace(0, magnitude.size, bucket_count + 1).astype(np.int64)
    peaks = np.zeros(bucket_count, dtype=np.float64)
    for i in range(bucket_count):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        if lo < magnitude.size:
            peaks[i] = magnitude[lo:hi].max()
    return peaks
