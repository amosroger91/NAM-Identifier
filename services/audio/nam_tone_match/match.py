# Tone matching utilities

import numpy as np
from typing import Dict, List, Tuple
from .peaks import compute_peaks


def compute_similarity(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Compute similarity between two audio signals.
    
    Args:
        signal1: First audio signal
        signal2: Second audio signal
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize signals
    signal1 = signal1 / (np.linalg.norm(signal1) + 1e-8)
    signal2 = signal2 / (np.linalg.norm(signal2) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(signal1, signal2) / (np.linalg.norm(signal1) * np.linalg.norm(signal2))
    
    # Ensure similarity is between 0 and 1
    return max(0, min(1, (similarity + 1) / 2))


def match_tone(input_audio: np.ndarray, reference_audio: np.ndarray, sample_rate: float) -> Dict:
    """
    Match input audio to reference audio.
    
    Args:
        input_audio: Input audio signal
        reference_audio: Reference audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary containing match results
    """
    # Find peaks in both signals
    input_peaks = compute_peaks(input_audio, int(sample_rate))
    reference_peaks = compute_peaks(reference_audio, int(sample_rate))
    
    # Compute similarity
    similarity = compute_similarity(input_audio, reference_audio)
    
    return {
        'similarity': similarity,
        'input_peaks': input_peaks,
        'reference_peaks': reference_peaks,
        'sample_rate': sample_rate
    }


def find_best_match(input_audio: np.ndarray, reference_library: List[Tuple[np.ndarray, str]], sample_rate: float) -> Tuple[str, float]:
    """
    Find the best matching reference from a library.
    
    Args:
        input_audio: Input audio signal
        reference_library: List of (audio_signal, name) tuples
        sample_rate: Sample rate of the audio
        
    Returns:
        Tuple of (best_match_name, similarity_score)
    """
    best_match = None
    best_similarity = -1
    
    for ref_audio, name in reference_library:
        result = match_tone(input_audio, ref_audio, sample_rate)
        if result['similarity'] > best_similarity:
            best_similarity = result['similarity']
            best_match = name
            
    return (best_match, best_similarity)


def make_clean_di(sample_rate: int, duration: float):
    """A few plucked notes: decaying sines at 196, 247, and 294 Hz."""
    rate = int(sample_rate)
    count = int(rate * float(duration))
    if count < 1:
        return np.zeros(0, dtype=np.float32)
    timeline = np.arange(count, dtype=np.float64) / rate
    mix = np.zeros(count, dtype=np.float64)
    for index, freq in enumerate((196.0, 247.0, 294.0)):
        start = int(index * 0.35 * rate)
        if start >= count:
            break
        elapsed = timeline[: count - start]
        mix[start:] += np.sin(2.0 * np.pi * freq * elapsed) * np.exp(-4.0 * elapsed)
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.8).astype(np.float32)


def _render_fixture(clean, effect):
    audio = np.asarray(clean, dtype=np.float64)
    kind = effect.get("type")
    if kind == "lowpass":
        kernel = np.ones(8, dtype=np.float64) / 8.0
        return np.convolve(audio, kernel, mode="same")
    if kind == "gain":
        return audio * 0.5
    return np.array(audio, copy=True)


def _log_magnitude(samples):
    audio = np.asarray(samples, dtype=np.float64)
    window = 1024
    hop = 256
    if audio.size < window:
        return np.log(np.abs(np.fft.rfft(audio)) + 1e-8)
    hann = np.hanning(window)
    frames = []
    for start in range(0, audio.size - window + 1, hop):
        frame = audio[start : start + window] * hann
        frames.append(np.log(np.abs(np.fft.rfft(frame)) + 1e-8))
    return np.concatenate(frames)


def _spectral_distance(target, rendered):
    count = min(len(target), len(rendered))
    if count == 0:
        return 0.0
    left = _log_magnitude(np.asarray(target, dtype=np.float64)[:count])
    right = _log_magnitude(np.asarray(rendered, dtype=np.float64)[:count])
    width = min(left.size, right.size)
    return float(np.mean((left[:width] - right[:width]) ** 2))


def rank_effects(target, clean, effects, sample_rate):
    """Rank fixtures by log-magnitude STFT error. Lower distance is closer."""
    del sample_rate
    heard = np.asarray(target)
    source = np.asarray(clean)
    ranked = []
    for effect in effects:
        distance = _spectral_distance(heard, _render_fixture(source, effect))
        ranked.append({**effect, "distance": distance})
    ranked.sort(key=lambda item: (item["distance"], str(item.get("name") or "")))
    return ranked
