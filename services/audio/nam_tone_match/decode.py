# Audio decoding utilities

import numpy as np
import wave
import miniaudio
from typing import Tuple


def decode_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Decode a WAV file into float32 mono samples in -1..1 range.
    
    Args:
        filepath: Path to the WAV file
        
    Returns:
        Tuple of (samples, sample_rate)
    """
    with wave.open(filepath, 'rb') as wav_file:
        # Get audio parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()  # in bytes
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        
        # Read all frames
        audio_bytes = wav_file.readframes(frames)
        
        # Convert to numpy array based on sample width
        if sample_width == 1:
            # 8-bit samples
            dtype = np.uint8
            samples = np.frombuffer(audio_bytes, dtype=dtype)
            # Convert to float32 in range [-1, 1]
            samples = (samples.astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            # 16-bit samples
            dtype = np.int16
            samples = np.frombuffer(audio_bytes, dtype=dtype)
            # Convert to float32 in range [-1, 1]
            samples = samples.astype(np.float32) / 32768.0
        elif sample_width == 3:
            # 24-bit samples
            # Convert to int32 first (padded with zeros)
            samples = np.frombuffer(audio_bytes, dtype=np.uint8)
            # Reshape and convert to int32
            samples = samples.reshape(-1, 3)
            # Convert from 24-bit to 32-bit signed integer
            samples = np.array([
                (sample[0] | (sample[1] << 8) | (sample[2] << 16))
                if sample[2] < 128 else
                ((sample[0] | (sample[1] << 8) | (sample[2] << 16)) - 0x1000000)
                for sample in samples
            ], dtype=np.int32)
            # Convert to float32 in range [-1, 1]
            samples = samples.astype(np.float32) / 0x800000
        elif sample_width == 4:
            # 32-bit samples
            dtype = np.int32
            samples = np.frombuffer(audio_bytes, dtype=dtype)
            # Convert to float32 in range [-1, 1]
            samples = samples.astype(np.float32) / 0x80000000
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # If stereo, downmix by averaging channels
        if channels > 1:
            samples = samples.reshape(-1, channels)
            samples = np.mean(samples, axis=1)
            
        return samples, sample_rate


def decode_mp3(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Decode an MP3 file into float32 mono samples in -1..1 range.
    
    Args:
        filepath: Path to the MP3 file
        
    Returns:
        Tuple of (samples, sample_rate)
    """
    # Use miniaudio to decode
    try:
        # Decode MP3 with miniaudio
        decoded = miniaudio.decode_file(filepath, output_format=miniaudio.SampleFormat.F32)
        
        # Get samples and sample rate
        samples = np.array(decoded.samples, dtype=np.float32)
        sample_rate = decoded.sample_rate
        
        # If stereo, downmix by averaging channels
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels)
            samples = np.mean(samples, axis=1)
            
        return samples, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to decode MP3: {str(e)}")
