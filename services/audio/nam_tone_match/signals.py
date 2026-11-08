# Basic signal generators

import numpy as np


def pluck(freq, seconds, rate, rng, brightness=0.5):
    """Karplus-Strong plucked string (dry, DI-like)."""
    count = int(np.ceil(seconds * rate)) + 1
    period = max(2, int(round(rate / freq)))
    buf = rng.uniform(-1.0, 1.0, period)
    for _ in range(int((1.0 - brightness) * 4)):  # soften the pick noise for gentler plucks
        buf = 0.5 * (buf + np.roll(buf, 1))
    out = np.empty(count)
    for i in range(count):
        j = i % period
        out[i] = buf[j]
        buf[j] = 0.4985 * (buf[j] + buf[(j + 1) % period])
    return out
