# Clean and dirty tone matching

The matcher first estimates drive from frame-level loudness variation, high-to-low frequency balance, and crest factor. It then adjusts the spectral profile comparison and applies a bounded penalty when a capture's measured distortion disagrees with the clip. The matcher measures distortion at each input trim using a two-tone probe. It keeps all captures eligible.

## Results

Both methods used the app's six-trim production index and the same 17 captures. Top-3 is the percentage of trials that returned the exact capture in the first three results.

| Query set | Trials | Previous top-1 | New top-1 | Previous top-3 | New top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three held-out performed DI recordings | 306 | 35.0% | 40.5% | 52.3% | 65.0% |
| Performed DI, clearly clean probe response | 48 | 0.0% | 45.8% | 0.0% | 72.9% |
| Performed DI, clearly dirty probe response | 234 | 40.6% | 38.5% | 63.2% | 65.0% |
| Black sampled-note guitar | 510 | 38.8% | 46.3% | 65.9% | 74.9% |
| Black sampled notes, clearly clean | 80 | 2.5% | 43.8% | 13.8% | 87.5% |
| Black sampled notes, clearly dirty | 390 | 44.6% | 44.1% | 76.9% | 71.8% |

On black sampled notes, the gain was +9.0 percentage points in top-3, with a 95% paired phrase-bootstrap interval of +1.6 to +14.9 points. On performed DI the gain was +12.7 points across three source recordings. The separate changes were +14.7, +12.7, and +10.8 points. Three recordings support a small stress test, so the performed-DI result omits a confidence interval.

The app gave a definite correct clean label on 62.5% of the 48 clear clean reamps, and a definite correct dirty label on 69.2% of the 234 clear dirty reamps. It marked 28.4% of those clear trials uncertain. The reference classes come from the capture's two-tone distortion at the trial's trim. These classes are a proxy for audible drive. Full songs, separation artifacts, and capture families outside the gallery were outside the test.

## Calibration and reproduction

The first sampled-note calibration called every clearly clean reamp of the bundled guitar take dirty. The final fit balances clean and dirty examples within both domains and gives performed reamps 70% of the training weight. It uses sampled-note and synthetic bank features plus three Guitar-TECHS takes. The fit includes the bundled take 02 excerpt and the first eight seconds of direct input takes 05 and 06. The held-out performances were the first eight seconds of Guitar-TECHS take 03 and GuitarJam test tracks 026 and 050. Each performed file contributed two non-overlapping four-second passages reamped at −6, 0, and +6 dB for evaluation. [Guitar-TECHS](https://zenodo.org/records/14963133) · [GuitarJam](https://huggingface.co/datasets/Julian-br/GuitarJam)

Use the [frozen capture and source-note setup](MEASUREMENT_RESULTS.md) and its four-trim render cache. Build the production index with the green guitar `ord` and `stac` directories in `NAM_DI_BANK`. The Guitar-TECHS DI tracks are in `P3_music/audio/directinput/`. Convert them from stereo 24-bit to mono 16-bit PCM by averaging the channels. The calibration stores the prepared WAV hashes. [Calibration](../services/audio/nam_tone_match/drive_calibration.json) · [Metrics](drive_matching_metrics.json)

```powershell
python scripts/measure_drive.py calibrate --di data/drive-techs-05.wav --di data/drive-techs-06.wav
python scripts/measure_drive.py black --index-data data
python scripts/measure_drive.py recorded --index-data data --di data/drive-techs-03.wav --di data/drive-jam-026.wav --di data/drive-jam-050.wav
```

The verification suite passed 8 Node tests and 49 Python tests. Browser verification confirmed that the tone estimate appears in match results and the dry/NAM switch shares the audition playhead.
