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


def make_clean_di(sample_rate: int, duration: float) -> List[float]:
    """
    Create a clean DI signal for testing.
    
    Args:
        sample_rate: Sample rate of the audio
        duration: Duration in seconds
        
    Returns:
        List of audio samples
    """
    # Generate a simple clean DI signal (sine wave with some noise)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Create a sine wave at 100 Hz with some noise
    clean_di = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(len(t))
    return clean_di.tolist()


def rank_effects(target: np.ndarray, reference: np.ndarray, effects: List[Dict], sample_rate: float) -> List[Dict]:
    """
    Rank effects based on their impact on the audio.
    
    Args:
        target: Target audio signal
        reference: Reference audio signal (clean DI)
        effects: List of effect dictionaries
        sample_rate: Sample rate of the audio
        
    Returns:
        List of effect dictionaries ranked by impact
    """
    ranked_effects = []
    
    for effect in effects:
        # Apply the effect to reference signal
        if effect.get('type') == 'lowpass':
            # Apply lowpass filter using the existing function from effects.py
            try:
                from .effects import apply_lowpass_filter
                processed_reference = apply_lowpass_filter(reference, sample_rate)
            except ImportError:
                # Fallback to simple convolution if the function is not available
                kernel = np.ones(8, dtype=np.float32) / 8
                processed_reference = np.convolve(reference, kernel, mode='same')
        elif effect.get('type') == 'identity':
            # Identity effect - no change
            processed_reference = reference.copy()
        elif effect.get('type') == 'gain':
            # Apply gain adjustment
            processed_reference = reference * 0.5  # Simple gain reduction
        else:
            # Unknown effect - no change
            processed_reference = reference.copy()
        
        # Compute similarity between processed reference and target
        similarity = compute_similarity(processed_reference, target)
        
        # Impact is how similar the processed reference is to the target
        # (higher similarity means better match)
        ranked_effects.append({**effect, 'impact': similarity})
    
    # Sort by impact (descending) - higher similarity means better match
    ranked_effects.sort(key=lambda x: x['impact'], reverse=True)
    
    return ranked_effects