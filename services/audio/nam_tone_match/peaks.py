# Peak detection utilities

import numpy as np
from typing import List


def compute_peaks(audio: np.ndarray, sample_rate: int, bucket_count: int = 1000) -> np.ndarray:
    """
    Compute peaks from audio data.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        bucket_count: Number of buckets to use (default 1000)
        
    Returns:
        Array of peaks in range [0, 1]
    """
    if len(audio) == 0:
        return np.array([])
    
    # Create buckets
    bucket_size = max(1, len(audio) // bucket_count)
    
    # Compute peak for each bucket (max absolute value in the bucket)
    peaks = []
    for i in range(0, len(audio), bucket_size):
        end_idx = min(i + bucket_size, len(audio))
        bucket_data = audio[i:end_idx]
        peak = np.max(np.abs(bucket_data))
        peaks.append(peak)
        
    # Ensure exactly bucket_count peaks by adjusting the last bucket
    if len(peaks) > bucket_count:
        # If we have too many buckets, merge the last few
        while len(peaks) > bucket_count:
            # Merge last two buckets
            last_peak = peaks.pop()  # Remove last peak
            second_last_peak = peaks[-1]  # Get second to last
            # Take the maximum of both peaks
            peaks[-1] = max(second_last_peak, last_peak)
    elif len(peaks) < bucket_count:
        # If we have too few buckets, pad with zeros or extend last bucket
        # This is a simple approach - just add zeros to reach exact count
        while len(peaks) < bucket_count:
            peaks.append(0.0)
    
    # Convert to numpy array and normalize to [0, 1] range
    peaks_array = np.array(peaks)
    
    # Normalize to 0..1 range (if all peaks are the same, keep them as 0)
    if np.max(peaks_array) > 0:
        peaks_array = peaks_array / np.max(peaks_array)
    
    return peaks_array
