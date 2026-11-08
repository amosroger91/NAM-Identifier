# Synthetic guitar phrases used to characterise each capture

import numpy as np

from .signals import pluck

RATE = 48000


def make_phrase(seed: int, seconds: float = 6.0, rate: int = RATE):
    """Random plucked-note phrase: notes across the neck, varied picking strength and rhythm."""
    rng = np.random.default_rng(seed)
    total = int(seconds * rate)
    mix = np.zeros(total)
    pos = 0
    while pos < total - rate // 2:
        note = int(rng.choice([40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64, 67, 69]))
        level = float(rng.uniform(0.3, 1.0))
        length = min(int(rng.uniform(0.3, 0.9) * rate), total - pos)
        string = pluck(440.0 * 2 ** ((note - 69) / 12), length / rate, rate, rng, 0.3 + 0.6 * level)
        mix[pos:pos + length] += string[:length] * level
        pos += int(rng.uniform(0.2, 0.5) * rate)
    return (mix / (np.max(np.abs(mix)) or 1.0) * 0.6).astype(np.float32)


def color_di(audio, seed: int, rate: int = RATE):
    """Random 'different guitar / pickup / room': spectral tilt, treble rolloff, bass shelf and a broad
    resonance. Training on many of these teaches the matcher to ignore the guitar and listen to the amp."""
    rng = np.random.default_rng(seed + 77)
    audio = np.asarray(audio, dtype=np.float64)
    spectrum = np.fft.rfft(audio)
    f = np.fft.rfftfreq(len(audio), 1.0 / rate)
    octaves = np.log2(np.maximum(f, 50) / 1000)
    db = rng.uniform(-9, 4) * octaves
    fc = rng.uniform(2500, 12000)
    db -= 10 * np.log10(1 + (f / fc) ** rng.choice([2, 4]))
    db += rng.uniform(-4, 4) * np.exp(-np.maximum(octaves + 2.0, 0) ** 2)
    center, width, gain = rng.uniform(1200, 5000), rng.uniform(0.5, 1.5), rng.uniform(-4, 6)
    db += gain * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / center) / width) ** 2)
    out = np.fft.irfft(spectrum * 10 ** (db / 20), len(audio))
    return (out / (np.max(np.abs(out)) or 1.0) * 0.6).astype(np.float32)


def training_phrases(count: int, seconds: float = 6.0, rate: int = RATE):
    """`count` deterministic, differently coloured phrases."""
    return [color_di(make_phrase(9001 + i, seconds, rate), 9001 + i, rate) for i in range(count)]
