# NAM Tone Match

Find Neural Amp Modeler captures that resemble a guitar tone in a song. Upload an MP3 or WAV, select a passage, isolate its guitar, and audition ranked captures in a local web app.

**Working prototype.** The matcher estimates whether the isolated guitar is clean or overdriven, then ranks captures by tone and drive. Audition the same recorded guitar take through each result and switch between the dry and processed signals while it plays.

[Get started](#get-started) · [Capture collection](#capture-collection) · [Matching method](#how-matching-works) · [Measured accuracy](#measured-accuracy) · [Troubleshooting](#troubleshooting)

## What the app does

1. Accepts a local MP3 or WAV and displays its waveform.
2. Lets you select a time range with a pointer or keyboard.
3. Uses Demucs `htdemucs_6s` to extract the guitar stem from that passage.
4. Estimates drive and ranks the `.nam` captures in the active model directory.
5. Plays a recorded guitar take through each candidate with a dry/NAM toggle and offers the original `.nam` for download.
6. Describes the isolated guitar (clean, edge of breakup, crunch, overdrive, high gain, or fuzz, plus words such as bright, dark, and saturated).
7. Searches the library by tone: the same dry phrase is rendered through each capture and scored against the clip.

Audio processing runs on your computer. The API binds to `127.0.0.1`; no cloud processing service or account is configured. Installing dependencies, downloading separator weights, and collecting captures require internet access.

## Get started

The commands below are for Windows PowerShell, run from the repository root. Development has been checked with **Python 3.12.10, Node.js 24.13.0, and npm 11.6.2**. Python dependencies are specified in [requirements.txt](requirements.txt); the frontend has a committed npm lockfile.

### 1. Install dependencies

Use a project-local Python environment and put it first on this terminal's `PATH`. This matters because the Node API launches the command named `python`.

```powershell
python -m venv .venv
$env:PATH = "$(Join-Path $PWD '.venv\Scripts');$env:PATH"
python -m pip install -r requirements.txt
npm.cmd --prefix apps/web ci
npm.cmd --prefix apps/web run build
```

`npm.cmd` works in PowerShell without relying on the execution policy for npm's `.ps1` wrapper. Repeat the `PATH` line in a new terminal before starting the API or running Python checks. If you already have a suitable environment, use it consistently instead.

### 2. Add captures

Put a few `.nam` files directly in `models/`, or use the [collection tools](#capture-collection). The app scans that directory's top level; it does not recursively scan folders or read `nams/` directly.

A fresh clone includes the app, the licensed audition recording, the collection manifests, and benchmark results. Capture weights, user uploads, and local caches are gitignored.

### 3. Start the app

```powershell
npm.cmd start
```

Open [http://127.0.0.1:3210/](http://127.0.0.1:3210/). The API serves the built frontend as well as the audio endpoints.

The first isolation downloads the Demucs weights if needed. The first match renders the phrase bank at six input trims for every uncached capture. On CPU, start with a small folder or run `python scripts/index_library.py --models=path\to\captures` before matching. Subsequent matches use the index in `data/`.

### 4. Match a passage

- Upload a song and select a passage with a prominent, consistent guitar sound.
- Click **Isolate guitar**, then listen to the extracted stem.
- Click **Match NAM** to see the estimated drive and ranked captures.
- Use **Play** to hear a recorded electric guitar performance through the matched capture. Click **NAM effect** during playback to switch between the capture and dry guitar at the same point in the take. **Download NAM** saves the original capture file.

On the waveform, arrow keys move the range start; Shift + arrow keys move its end, in one-second steps. Invalid or out-of-bounds ranges are rejected.

The isolated-guitar player plays the stem from your song. The audition player uses the app's dry reference take. After changing the range, isolate the new selection before matching.

## Capture collection

The collector downloads public NAM files from the checked-in source manifests, validates their JSON structure, deduplicates identical file contents with SHA256, and records their provenance.

The local collection assembled during development contained **1,602 distinct NAM files, about 517 MB**, including clean/crunch/high-gain amps, bass gear, pedals, and gain/cabinet variations. This is a snapshot of local files, not a bundled dataset or an audio-quality certification. Download availability and counts can change.

### Download repository collections

```powershell
python scripts/collect_nams.py --config scripts/nam_sources.json --output nams --mirror-dir models
```

### Download supplemental captures

The supplemental manifest includes compressed NAMZ files. Install the optional official decoder in the same Python environment first:

```powershell
python -m pip install namz==1.3.0
python scripts/collect_nams.py --config scripts/nam_extra_sources.json --output nams --mirror-dir models
```

NAMZ files are decoded to standard `.nam` JSON. The catalog retains both the source and decoded hashes. Ordinary NAM collection needs only Python's standard library; NAM rendering uses the audio dependencies above.

Both commands can be rerun to resume an import. `--mirror-dir models` makes captures available to the app through hardlinks, with a copy fallback when hardlinks are unavailable. Hardlinks share the same file contents, so edit neither copy if you want to preserve the recorded hash.

### Limits and reports

| Option | Default | Meaning |
| --- | --- | --- |
| `--workers` | `4` | Concurrent file downloads; allowed values are 1–4. |
| `--max-files` | `5000` | Maximum new library files in one run. |
| `--max-bytes` | `5368709120` | Per-run ceiling for network bytes and stored capture bytes, each 5 GiB. |
| `--max-file-bytes` | `134217728` | Maximum individual capture size, 128 MiB. |
| `--dry-run` | off | Discover URLs without saving captures; still uses remote metadata APIs. |

Run `python scripts/collect_nams.py --help` for all options. Inspect the summary even when the process exits successfully: an import can complete partially while recording individual errors.

| Local artifact | Contents |
| --- | --- |
| `nams/catalog.jsonl` | Original URLs, hashes, file sizes, model metadata, source terms, and conversion details. |
| `nams/summary.json` | Results and errors from the most recent collector run. |
| `nams/sources/` | Supporting creator/license documents retained during development. |
| `nams/LIBRARY_SUMMARY.json` | Development inventory and structural validation snapshot, when present. |

The collector supports public GitHub repositories, direct files, and archives. GitHub API rate limits may pause repository discovery; completed imports remain available for a later retry. Source manifests contain download references, not a blanket license for the models. Preserve each creator's terms and attribution; some sources have incomplete or custom terms. The downloaded collection stays gitignored.

## How matching works

The production path uses separate drive and tone comparisons.

1. Decode the upload, downmix to mono, and cut the selected range.
2. Run Demucs `htdemucs_6s` and retain its `guitar` stem.
3. Estimate drive from loudness variation, frequency balance, and crest factor. Display **Clean**, **Dirty / overdriven**, or **Uncertain / edge of breakup**.
4. Compare the stem's spectrum with each capture's cached profiles, using the drive estimate to adjust band weights and the contribution of the recorded-note centroid. Apply a bounded penalty when a capture's measured drive at that input trim differs from the estimate. Every capture remains eligible.

The index renders eight synthetic phrases and, when `NAM_DI_BANK` is configured, eight assembled recorded-note phrases at −12, −6, 0, +6, +12, and +18 dB of digital input trim. It stores level-normalized quarter-octave profiles between roughly 90 Hz and 10 kHz. A separate two-tone probe measures each capture's distortion across the same trims.

Match distance combines spectral differences with the drive penalty. Lower values rank higher. `probeInputTrimDb` gives the selected prototype's digital input trim. The result also reports `searched`, `ranked`, `skipped` (captures that could not be rendered, with the reason), `bank` (the phrase counts), and `drive` (the estimated class and any candidates whose drive could not be measured).

**Recorded-guitar bank.** Set `NAM_DI_BANK` to one or more folders (separated by `;` on Windows) of dry electric-guitar clips, ideally single notes:

```powershell
$env:NAM_DI_BANK = "C:\path\to\dry\guitar\notes"
```

Without it the bank is synthetic only, and the UI flags that configuration. Pre-build indexes with the `index` command or `scripts/index_library.py` to avoid rendering every capture on the first match.

Real NAM files run through the Python NAM inference library. The loader converts legacy (v0.5) WaveNet configs, which renders bit-identically to an older reference release on the 16 captures tested (`scripts/render_conformance.py`), and selects the largest submodel of `SlimmableContainer` captures (no independent reference exists for those).

`rank_effects` (aligned STFT error) remains only for the fixture tests. A warmed Python worker renders the recorded DI through each capture and caches the result in `data/`. The audition player uses one playhead for the dry and processed signals and matches their RMS playback levels.

The bundled guitar performance is the 12–20 second excerpt of `P3_music/audio/directinput/directinput_02.wav` from Guitar-TECHS by Hegel Emmanuel Pedroza Villalobos, Termeh Taheri, Wallace Abreu, Ryan Corey, and Iran R. Roman. The excerpt was converted to mono 16-bit PCM with a short fade-in. [Source dataset](https://zenodo.org/records/14963133) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Set `NAM_AUDITION_DI` to a dry guitar WAV or MP3 before starting the app to audition your own performance. Every capture uses up to eight seconds of that take.

## Search by tone

The app already renders audio through `.nam` files. Tone search uses that. It is a separate listening search from the identity ranker above: Match NAM still asks which capture produced this spectrum. Search by tone asks which captures sound like this clip.

The app describes the guitar automatically after isolation. **Search by tone** reuses those measurements for the same isolation revision. It compares drive, brightness, mids, attack, dynamics, and spectrum with measurements from a shared phrase rendered through each capture. Low-confidence drive estimates receive less weight. The top result shows how many points each component contributed to the listening score.

Cards are cached in `data/tone-v1`. The first search of a large library renders every uncached capture. Build them ahead of time with:

```powershell
python scripts/index_library.py --tone --models=models
```

A stem from a full mix still carries room, delay, and separation leftovers. A dry guitar recording is the better reference. CLAP is optional and off by default. It adds an audio embedding and scores the clip against short phrases such as "crunchy electric guitar". It does not replace the measurements.

```powershell
python -m pip install transformers
$env:NAM_CLAP = "1"
```

`NAM_CLAP_MODEL` can point at a local checkout of `laion/clap-htsat-unfused`. The first enabled run downloads that model. Set `NAM_CLAP=0` to force the signal-only path.

`scripts/tone_benchmark.py` scores the gain label, adjectives, and search self-retrieval over every capture in a folder, using seen and unseen guitar phrases, dry and with plate reverb. The baseline found that the old drive estimate read most recorded-guitar phrases as clean. Describe tone now uses a separate phrase gain model (`gain_calibration.json`, refit with `scripts/fit_gain_model.py`) that estimates the same sine-probe THD the tone cards record. Bright and dark come from the 2–8 kHz spectral tilt (dB per octave), which follows the capture rather than which notes were played; the card offset and thresholds are refit with `scripts/fit_tilt.py`. [Tone benchmark results](benchmarks/TONE_BENCHMARK_RESULTS.md)

Two files can be compared without the library:

```powershell
Set-Location services/audio
'{"command":"compare","a":"C:/path/a.wav","b":"C:/path/b.wav"}' | python -m nam_tone_match
```

## Measured accuracy

Top-3 means the exact capture appeared in the first three results. Both methods used the same 17 captures and the app's six-trim index.

| Test | Trials | Previous top-3 | Drive-aware top-3 |
| --- | ---: | ---: | ---: |
| Held-out performed guitar recordings | 306 | 52.3% | 65.0% |
| Clearly clean subset of those recordings | 48 | 0.0% | 72.9% |
| Black guitar sampled-note stress test | 510 | 65.9% | 74.9% |

Clean retrieval improved most. On the performed recordings, dirty-tone top-3 rose from 63.2% to 65.0%. On the sampled-note stress test it fell from 76.9% to 71.8%. The report covers the drive labels, separate takes, and reproduction steps. [Drive matching results](benchmarks/DRIVE_MATCHING_RESULTS.md)

## Current limits

The central difficulty is comparing different performances. Guitar/pickup response, notes, picking, cabinet/microphone coloration, mix processing, and separation artifacts all affect the measured spectrum. Normalizing level does not remove these differences. Clean captures and nearby settings can also be hard to distinguish from wet audio alone.

Current limitations include:

- Full-library retrieval and real-song accuracy have not been established. Passing unit tests and loading a model's JSON do not demonstrate perceptual accuracy.
- Original stereo information is discarded before separation. Doubled or differently panned guitars can yield an ambiguous stem.
- All captures are ranked independently; there is no family grouping, calibrated rejection, or “no suitable match” decision.
- Cabinet IR search and automatic corrective EQ are planned.
- Similar settings on the same amp can still trade places in the ranking.
- A capture that fails to render is skipped and reported, but a cold-cache search of a large library is slow and has no progress display or cancellation. Index first.
- The drive benchmark covers three held-out recordings reamped through a fixed capture set. The sampled-note stress tests use a library with an unverified recording signal path. Accuracy on Demucs-separated song clips requires separate evaluation.

[ACCURACY_PLAN.md](ACCURACY_PLAN.md) sets out the next stages: reproduce a frozen small benchmark, introduce verified real DIs, compare features on held-out instruments, fit compatible amp/cabinet chains, test realistic mixes, and use blinded listening to decide whether sound quality improved.

## Validation

### Project gate

```powershell
powershell -NoProfile -File .\scripts\verify.ps1
```

Success prints a line exactly equal to `VERIFY_OK` and exits with status 0. If PowerShell's execution policy prevents this local script from running, a process-scoped invocation is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

The gate runs Node API/range tests, Python unit tests, and fixture behavior checks. Heavy separation and matching calls in API tests are stubbed. It does not download Demucs weights, exercise the browser UI, or certify a real model collection. Build the frontend separately with `npm.cmd --prefix apps/web run build`.

### Retrieval experiment

Use a deliberately selected, small directory of captures, with files directly inside it:

```powershell
python scripts/accuracy_check.py path\to\small-model-set
python scripts/accuracy_check.py path\to\small-model-set --samples="path\to\dry-guitar-wavs"
```

The script makes targets by rendering held-out synthetic phrases through each capture at four gains; the optional sample directory adds recorded-note phrases. It reports top-1, top-3, mean reciprocal rank, mean rank, and frequent confusions. Its current success threshold is aggregate top-1 of at least 90%.

This is an experimental harness: it does not separate realistic mixes, enforce instrument/session splits, or currently print separate aggregate scores for synthetic and recorded cases. A full-library run can be very slow. See the accuracy plan before using its results as a generalization claim.

### Earlier controlled benchmark

The controlled benchmark uses `scripts/measure_accuracy.py`, a hash-checked set of 17 captures, and a frozen source-note split. It uses green guitar queries to select the challenger and black guitar queries for the final comparison. Every method searches the same four input trims. The sample library's recording signal path is unverified.

| Corrected black guitar experiment | Top-1 | Top-3 | MRR |
| --- | ---: | ---: | ---: |
| Original single synthetic probe | 24.1% | 42.2% | 0.395 |
| Recorded-note bank, mean-power centroid | 46.9% | 75.3% | 0.635 |
| Recorded + synthetic bank, nearest prototype | 40.2% | 67.6% | 0.570 |

Both bank methods passed the top-3 gate. The [results report](benchmarks/MEASUREMENT_RESULTS.md) includes paired intervals, limitations, and the exact reproduction steps.

```powershell
python scripts/measure_accuracy.py --render --samples "path\to\Samples" --workers 3 --threads 1
python scripts/measure_accuracy.py --final --samples "path\to\Samples"
```

`--final` reads the checked-in selection and checks the source fingerprints. A fresh experiment can create a manifest with `--freeze` and save its selection with `--explore`. The cache lives under `data/bench-cache-v2/`.

The earlier `scripts/run_bench.py` experiment used unequal search trims and sampled some source WAVs more than once. Its reported scores describe 510 reamp trials from ten assembled query phrases on one black guitar sample set.

| Earlier black guitar experiment | Top-1 | Top-3 | MRR |
| --- | ---: | ---: | ---: |
| Previous method (single synthetic probe) | 23.5% | 40.6% | 0.387 |
| Recorded+synthetic bank, nearest prototype | 39.6% | 60.4% | 0.544 |
| Paired change (95% CI) | +16.1 (+11.4, +20.8) | +19.8 (+14.9, +24.7) | +0.16 |

The earlier results preceded the corrected comparison. Real-song accuracy requires separate evaluation with known mixtures and recordings from independent sessions.

### Legacy renderer diagnostic

The conformance helper compares legacy v0.5 WaveNet renders with a separate NAM 0.10.0 installation. Keep that reference installation separate from the app's runtime and use a small legacy-only set:

```powershell
python -m pip install --no-deps --target data/oldnam neural-amp-modeler==0.10.0
python scripts/render_conformance.py path\to\legacy-model-set --old=data/oldnam
```

The reference still needs compatible supporting dependencies in the active environment. This diagnostic checks renderer agreement on its chosen probe; it does not measure tone retrieval or certify every NAM architecture.

## Development and architecture

The stack is React/Vite, Node's built-in HTTP server, and a Python/NumPy/PyTorch audio package. There is no database or cloud host.

```text
Browser: React + Vite
        │ HTTP on 127.0.0.1:3210
Node API: apps/api
        │ JSON subprocess bridge and persistent audition worker
Audio package: services/audio/nam_tone_match
        ├── decode → slice → Demucs guitar stem
        ├── drive estimate → spectral comparison → ranking
        ├── recorded DI → level-matched dry/NAM audition
        └── local models/ and data/ artifacts
```

For frontend development, run the API and Vite in separate terminals with the same Python environment:

```powershell
# Terminal 1
npm.cmd start

# Terminal 2
npm.cmd --prefix apps/web run dev
```

Use the URL printed by Vite. Its `/api` proxy targets port 3210. `vite preview` alone does not start the audio API.

| Path | Purpose |
| --- | --- |
| `apps/web/` | Upload, waveform/range selection, stem playback, and ranked capture UI. |
| `apps/api/` | HTTP routes, local file storage, and Python subprocess bridge. |
| `services/audio/nam_tone_match/` | Decode, separation, NAM loading/rendering, profiles, and JSON CLI. |
| `scripts/collect_nams.py` | Resumable capture collection and provenance recording. |
| `scripts/accuracy_check.py` | Experimental retrieval benchmark. |
| `scripts/measure_accuracy.py` | Reproduce the frozen four-trim comparison. |
| `scripts/measure_drive.py` | Calibrate and evaluate the clean/dirty matcher. |
| `scripts/index_library.py` | Build tone and drive caches ahead of matching. |
| `scripts/render_conformance.py` | Legacy renderer comparison diagnostic. |
| `scripts/verify.ps1` | Required project verification gate. |
| `models/` | Flat active capture directory; model files are ignored. |
| `nams/` | Local collected library and catalog; ignored. |
| `data/` | Uploads, metadata, stems, generated probes, renders, and caches; ignored. |
| `SPEC.md` | Original build/test contract. Current production scoring is documented above. |
| `ACCURACY_PLAN.md` | Proposed accuracy work and evidence required to advance it. |

## HTTP API

The API returns JSON errors as `{"error":"..."}`. Uploads use raw bytes with an `x-filename` header. Capture names in playback URLs must be URL-encoded basenames present in that track's match results.

| Method | Route | Request / response |
| --- | --- | --- |
| `POST` | `/api/tracks` | Raw MP3/WAV → `{id, name, durationSec, sampleRate}`. |
| `GET` | `/api/tracks/:id/peaks` | Approximately 1,000 waveform peaks and duration. |
| `POST` | `/api/tracks/:id/isolate` | `{startSec, endSec}` → `{id, startSec, endSec, durationSec, revision, tone}`. Each isolation has its own description and revision. |
| `GET` | `/api/tracks/:id/guitar[?rev=N]` | Isolated guitar as WAV. A stale `rev` returns `409`. |
| `POST` | `/api/tracks/:id/match` | Optional `{revision}` (stale → `409`) → `{results:[{name, distance, probeInputTrimDb}], revision, searched, ranked, skipped, bank, drive}`. |
| `POST` | `/api/tracks/:id/describe` | Optional `{revision}` → the isolated guitar's tone description. |
| `POST` | `/api/tracks/:id/tone-search` | Optional `{revision}` → the query description and ranked results with similarity scores and their components. |
| `GET` | `/api/tracks/:id/rendered/:name[?rev=N]` | Audition WAV. With `compare=1`, the left channel carries the NAM output and the right carries the dry recording. |
| `GET` | `/api/tracks/:id/download/:name[?rev=N]` | Original matched `.nam` capture as an attachment. |

Missing/invalid uploads and ranges return `400`; matching before isolation returns `409`; missing Demucs returns `503`. Ranges must remain inside the source duration and are never silently clamped. Uploads are limited to 300 MiB.

Environment settings: `PORT` (default `3210`), `NAM_MODELS_DIR` (capture folder, default `models/`), `NAM_DI_BANK` (folders of dry guitar clips for the matching bank), and `NAM_AUDITION_DI` (optional dry guitar take for audition playback). There is no `.env` loader. If you change the API port during frontend development, also update the proxy in `apps/web/vite.config.js`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| “Web UI is not built” | Run `npm.cmd --prefix apps/web ci` and `npm.cmd --prefix apps/web run build` from the repository root. |
| “Demucs is not installed” | Install requirements with the same `python` the API finds on `PATH`, then restart the API. |
| MP3 decode fails with `_cffi_backend` missing | Check `python -m pip check` and `python -c "import miniaudio"`; repair/install `cffi` and `miniaudio` in that environment. WAV decoding uses the standard library. |
| NAM import fails or a DLL is blocked | Check the environment's PyTorch/NAM dependencies. The loader has an inference-only import fallback, but environment and architecture compatibility still matter. |
| No results | Confirm there are `.nam` files directly in `models/`; collecting into `nams/` alone requires the mirror option above. |
| First match appears slow | Uncached captures are being rendered synchronously. Start with a few captures to verify setup; inspect API errors if a capture fails. |
| Results remain stale after changing the renderer or bank | Indexes are keyed by capture content and bank; after changing the renderer or feature code, bump `FEATURE_VERSION` in `bank.py` or delete `data/index/`. |

Commit source code, reproducible benchmark evidence, and explicitly licensed app assets. The local library catalog preserves download sources and creator terms.
