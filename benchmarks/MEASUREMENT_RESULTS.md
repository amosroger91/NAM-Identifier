# NAM capture retrieval results

The corrected benchmark compares methods on 510 black-guitar reamp trials from ten assembled phrases, 17 frozen captures, and three query input trims. Each method searches the same four candidate input trims.

| Method | Top-1 | Top-3 | MRR | Top-3 change (95% paired interval) | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Original single synthetic probe, mean power | 24.1% | 42.2% | 0.395 | — | Baseline |
| Green recorded-note bank, mean-power centroid | 46.9% | 75.3% | 0.635 | +33.1 pp (+26.3 to +40.4) | Pass |
| Recorded + synthetic bank, mean-dB nearest prototype | 40.2% | 67.6% | 0.570 | +25.5 pp (+17.5 to +32.9) | Pass |

The gate required at least +10 percentage points in top-3 and a paired interval above zero. The bootstrap used 2,000 draws with seed 0 and resampled complete query-phrase groups. Each group kept all captures, input trims, and methods together.

## Development and failure cases

The recorded-note centroid led 16 challengers on green development queries at 78.2% top-3 and 0.701 MRR. The mixed nearest-prototype method scored 65.3% top-3 and the original probe scored 38.4%. The selection was saved before scoring black. On the synthetic diagnostic, the centroid scored 39.2% top-3, the original probe scored 81.8%, and the mixed nearest-prototype method scored 83.1%.

Clean captures still posed problems. With the centroid, Dirty Shirley Mini Clean reached 33.3% top-3, 5152 Clean-ish reached 40.0%, and Fender TwinVerb reached 56.7%. The centroid also dropped JCM2000 L2 G6 from 83.3% to 53.3% top-3, and JCM2000 Crunch from 93.3% to 90.0%. JCM2000 L2 G8 stayed at 100% top-3 while top-1 fell from 86.7% to 53.3%. The metrics file includes all 30 trials per capture. The app uses the mixed nearest-prototype method with six input trims. The table compares scoring methods using four.

This is a replication on the black sample set, which the earlier unequal-trim experiment also used. Across the green and black queries, 240 distinct source WAVs were used once each, with no overlap with the green bank. The sample recordings have an unverified signal path. The interval measures variation across assembled phrases in this library. Accuracy on independent players, recording sessions, and separated song mixes remains unmeasured by this experiment.

## Reproduce

Place the 17 files listed in [baseline17.json](baseline17.json) in `models/`. Pass the root of the Black and Green Guitars sample library to `--samples`. The [frozen protocol](measurement_protocol.json) contains source file hashes, note offsets, and the common candidate trims of −6, 0, +6, and +12 dB. The query trims are −6, 0, and +6 dB.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/measure_accuracy.py --render --samples "path\to\Samples" --workers 3 --threads 1
python scripts/measure_accuracy.py --final --samples "path\to\Samples"
```

`--final` reuses the saved [frozen selection](measurement_selection.json) and checks its protocol, code, and runtime fingerprints. To repeat challenger selection in a fresh experiment directory, use `--explore` between rendering and final evaluation with no existing selection file. `--freeze` creates a protocol for a fresh experiment.

All 2,686 required entries were validated. The six-worker attempt ran out of memory. The three-worker restart reused 316 completed entries and rendered the remaining 2,370 at about two renders per second over 19 minutes and 47 seconds. The corrected cache reused zero entries from the earlier experiment. The CPU runtime was Python 3.12.10, NumPy 2.4.6, PyTorch 2.12.0, and Neural Amp Modeler 0.13.0, with one Torch thread per worker.

`scripts/verify.ps1` returned `VERIFY_OK` with 8 Node tests, 46 Python tests, and the behavior check. The [aggregate metrics](measurement_metrics.json) include per-capture results. The [trial records](measurement_trials.json) include the capture hash, input trim, scores, and ranks for each trial.
