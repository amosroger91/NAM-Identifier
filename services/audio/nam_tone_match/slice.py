# Audio slicing utilities

import numpy as np


def slice_audio(audio: np.ndarray, start_sec: float, end_sec: float, sample_rate: int) -> np.ndarray:
    """Cut samples between two times in seconds. Never clamps: bad ranges raise ValueError."""
    if not (0 <= start_sec < end_sec):
        raise ValueError("Inverted range: start must be less than end")
    start_sample = int(round(start_sec * sample_rate))
    end_sample = int(round(end_sec * sample_rate))
    if start_sample >= end_sample:
        raise ValueError("Inverted range: start must be less than end")
    if end_sample > len(audio):
        raise ValueError(f"Range extends past end of audio: {end_sample} > {len(audio)}")
    return audio[start_sample:end_sample]
