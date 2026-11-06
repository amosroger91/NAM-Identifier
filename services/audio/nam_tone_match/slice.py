# Audio slicing utilities

import numpy as np
from typing import Tuple


def slice_audio(audio: np.ndarray, start_sec: float, end_sec: float, sample_rate: int) -> np.ndarray:
    """
    Cut a sample range from start and end seconds.
    
    Args:
        audio: Audio data as numpy array
        start_sec: Start time in seconds
        end_sec: End time in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        Array of samples for the specified range
        
    Raises:
        ValueError: If range is inverted or past the end of audio
    """
    # Convert seconds to sample indices
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    # Check for inverted range
    if start_sample >= end_sample:
        raise ValueError("Inverted range: start must be less than end")
    
    # Check if range is past the end of audio
    if end_sample > len(audio):
        raise ValueError(f"Range extends past end of audio: {end_sample} > {len(audio)}")
    
    # Return the slice
    return audio[start_sample:end_sample]
