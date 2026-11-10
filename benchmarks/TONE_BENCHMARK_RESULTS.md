# Tone description benchmark

`scripts/tone_benchmark.py` measures **Describe tone** and **Search by tone** the way the app runs them on an isolated stem.

## Setup

- **Captures.** The 40 captures in `data/test-models`.
- **Phrases.** Each capture is rendered through nine phrases at 0 dB trim, with the output normalised to a 0.5 peak:
  - `audition`: the DI phrase the drive model was fit on (reported as **seen**).
  - **Unseen:** three green-guitar and three black-guitar phrases assembled from recorded notes, plus two synthetic plucked phrases.
- **Reverb.** Every render is scored dry, with a 20% plate and with a 40% plate, for 1,080 clips in total.
- **Gain truth.** The capture's tone card, which comes from its sine-probe distortion reading.
- **Search truth.** The capture itself. A good search ranks it first out of 40.
- **CLAP** was off, which is the default.

To reproduce (renders are cached, so re-runs only re-describe):

```powershell
python scripts/tone_benchmark.py --models data/test-models
```

Per-clip rows go to `data/tone-bench-rows.json`. The summary goes to [`tone_benchmark.json`](tone_benchmark.json).

## Baseline (2026-11-09, before the phrase gain model)

### Gain label

| Group | Clips | Exact | Within one class | Bias (classes) |
| --- | ---: | ---: | ---: | ---: |
| Seen phrase, dry | 40 | 57% | 95% | +0.38 |
| Unseen phrases, dry | 320 | 20% | 31% | −1.99 |
| Unseen phrases, 40% plate | 320 | 20% | 31% | −1.91 |

On recorded-guitar phrases the drive model mostly answers **clean**:

| Phrase | Predicted clean |
| --- | ---: |
| `audition` | 5% |
| green guitar | 80–92% |
| black guitar | 72–98% |
| synthetic | 2–5% |

For example, 227 of 406 overdrive clips were called clean. Plate reverb barely changes the result.

### Tone search (rank of the true capture among 40)

| Group | Top-1 | Top-5 | MRR | Top result is `darwinscat … Ampeg SGT DI` |
| --- | ---: | ---: | ---: | ---: |
| All | 11% | 30% | 0.23 | 47% of clips |
| Synthetic phrases | — | 75% | — | — |
| Recorded-guitar phrases | — | 12–28% | — | — |

One capture is the top result for nearly half of all queries.

### Adjectives

| Word | Query rate | Card rate | Agreement |
| --- | ---: | ---: | ---: |
| dark | 77% | 38% | 61% |
| compressed | 0% | 57% | 42% |
| saturated | 29% | 78% | 43% |
| mid-heavy | 52% | 57% | 77% |

- **Brightness.** Median query brightness is −15.4 dB, against −7.4 dB on the cards. Recorded phrases sit between −14 and −24 dB, and `audition` at −33 dB.
- **Not CLAP.** CLAP was off, so "dark" is not coming from CLAP.

## What this says

1. **The drive model doesn't generalise.** It works on the phrase it was fit on and on synthetic plucks, but reads real recorded notes as clean.
   - Its four dynamics features (level spread, spectral tilt, crest) depend more on the playing than on the amp.
   - It needs refitting on many phrases, recorded ones included, or features that are less sensitive to the input.
2. **"Dark" describes the phrase, not the amp.** `brightnessDb` is an absolute high-vs-low band balance, so a bassy performance reads as dark on any amp.
   - The fix is to measure brightness relative to the input, or relative to what a neutral capture does to the same phrase.
3. **Search inherits both problems.**
   - The cards are built from a synthetic phrase, and synthetic queries retrieve well (75% top-5).
   - Recorded-guitar queries land on whichever capture best fits a dark, low-drive vector, which is `darwinscat … SGT DI`.
4. **"Compressed" never fires on a query.** Its crest threshold is miscalibrated for real phrases.

**Caveat.** The green/black sample library's recording chain is unverified, so it may not be a dry DI. These numbers are a stress test, not a real-song accuracy figure.

## After the phrase gain model (2026-11-09)

What changed:

- **Describe tone has its own gain model.** On a phrase, it now uses [`gain_model.py`](../services/audio/nam_tone_match/gain_model.py) instead of slicing the clean/dirty drive probability into five classes.
  - That drive target saturates: 33 of the 40 test captures score 1.0 at 0 dB.
  - The new model predicts the tone card's sine-probe THD (log scale) from ten playing-insensitive features: short-term crest, level spread, sustain, spectral fill and tilt movement.
  - It is a ridge fit with a penalty that keeps dry and plate-reverb renders on the same prediction.
- **Training is kept apart from the benchmark.** `scripts/fit_gain_model.py` uses 63 captures that aren't in `data/test-models`, eight phrases (audition, green guitar seeds 7001–7004 and three coloured synthetic phrases), four input trims, and dry and plate renders.
  - The black guitar and the benchmark's phrase seeds are never used.
  - Cross-validated by capture: 36% exact and 80% within one class.
- **Search compares like with like.** Tone cards store the measured probe gain for search, and the query's estimate is compared against it (card version 3).
- **Match NAM is unchanged.** `drive_calibration.json` and its ranking are untouched.

Unseen phrases:

| | Exact | Within one class | Mean class distance | Bias |
| --- | ---: | ---: | ---: | ---: |
| Before, dry | 20% | 31% | 2.26 | −1.99 |
| After, dry | 28% | 54% | 1.23 | −1.15 |
| Before, 40% plate | 20% | 31% | 2.23 | −1.91 |
| After, 40% plate | 17% | 49% | 1.49 | −1.45 |

- **Recorded guitars aren't mostly "clean" any more.** Green and black phrases now read clean 18% of the time (truth: 8%), down from 72–98%.
- **High gain still reads lower.** The remaining negative bias is mostly high-gain clips read as crunch or overdrive. That's a linear model shrinking toward the middle on a capture set that is mostly high-gain.
- **Tone search** top-5 is 28% (was 31%). The most frequent top result fell from 47% to 35% of clips. Search is now limited by the spectrum and brightness terms, the "dark" problem, which this change doesn't touch.
