# NAM Tone Match

Find Neural Amp Modeler captures that resemble a guitar tone in a song. Upload an MP3 or WAV, select a passage, isolate its guitar, and audition ranked captures in a local web app.

**Status: working prototype; tone matching is experimental.** Audio decoding, guitar separation, NAM rendering, and the HTTP workflow are implemented. The current ranker uses a coarse spectral comparison against a synthetic guitar phrase. Treat its results as starting points for listening. Real-song accuracy and the full capture library have not been validated.

[Get started](#get-started) · [Capture collection](#capture-collection) · [Matching method](#how-matching-works) · [Validation](#validation) · [Accuracy roadmap](ACCURACY_PLAN.md)

## What the app does

1. Accepts a local MP3 or WAV and displays its waveform.
2. Lets you select a time range with a pointer or keyboard.
3. Uses Demucs `htdemucs_6s` to extract the guitar stem from that passage.
4. Compares the stem with renders of the `.nam` captures in `models/`.
5. Lists candidates by distance and plays a test phrase through each selected capture.

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

A fresh clone includes the application and source manifests. **Third-party capture weights, audio files, and local caches are not included in Git.**

### 3. Start the app

```powershell
npm.cmd start
```

Open [http://127.0.0.1:3210/](http://127.0.0.1:3210/). The API serves the built frontend as well as the audio endpoints.

The first isolation downloads the Demucs model weights if they are missing. Separation can take time on CPU. The first match also renders every uncached capture at four input trims; a large library makes that initial operation expensive. Subsequent matches reuse cached profiles.

### 4. Match a passage

- Upload a song and select a passage with a prominent, consistent guitar sound.
- Click **Isolate guitar**, then listen to the extracted stem.
- Click **Match NAM** to rank the active captures.
- Use **Play** beside a result to hear the synthetic test phrase through that capture at its selected input trim.

On the waveform, arrow keys move the range start; Shift + arrow keys move its end, in one-second steps. Invalid or out-of-bounds ranges are rejected.

Candidate playback uses a generated phrase. It does not replay the song's performance through the candidate. After changing the selected range, run **Isolate guitar** again before matching: automatic invalidation of the previous stem is still a planned fix.

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

The current production path is deliberately simple:

1. Decode the upload, downmix to mono, and cut the selected range.
2. Run Demucs `htdemucs_6s` and retain its `guitar` stem.
3. Generate a deterministic **12-second synthetic DI-like phrase** of plucked notes and chords at 48 kHz.
4. Render it through each capture at **−6, 0, +6, and +12 dB** of digital input trim.
5. Summarize each render and the reference as a level-normalized average spectrum in quarter-octave bands, approximately 90 Hz–10 kHz. Ignore very quiet frames.
6. Choose each capture's lowest RMS profile difference and sort by distance, breaking ties by filename.

Distance is an RMS difference between spectral profiles in dB. A lower value indicates a closer match under this particular feature representation. It is not a calibrated confidence score. The returned `inputGainDb` is the best tested digital trim for the synthetic probe; it does not identify the original artist's amp setting or recording level.

Real NAM files run through the Python NAM inference library. The loader includes a conversion for legacy WaveNet configs and selects the largest submodel in `SlimmableContainer` captures. It resamples the probe to the model's declared rate and returns the output at the analysis rate.

Profiles are computed on demand and saved in `data/profile-cache/`. The current cache key uses capture filename, size, modification time, and a feature-version number. After changing the probe, renderer, or analysis settings, remove the relevant cached profiles before comparing results; those inputs are not all encoded in the cache key yet.

The older `rank_effects` function remains for fixture tests using aligned STFT error. The app uses `rank_captures` and the averaged profiles described above. `data/clean-di.wav` is a separate two-second fixture signal; production ranking and auditioning use `data/probe-di.wav`.

## Accuracy and current limits

The central difficulty is comparing different performances. Guitar/pickup response, notes, picking, cabinet/microphone coloration, mix processing, and separation artifacts all affect the measured spectrum. Normalizing level does not remove these differences. Clean captures and nearby settings can also be hard to distinguish from wet audio alone.

Current limitations include:

- Full-library retrieval and real-song accuracy have not been established. Passing unit tests and loading a model's JSON do not demonstrate perceptual accuracy.
- Original stereo information is discarded before separation. Doubled or differently panned guitars can yield an ambiguous stem.
- All captures are ranked independently; there is no family grouping, calibrated rejection, or “no suitable match” decision.
- Cabinet IR search, user-supplied DI audition, automatic corrective EQ, and level-matched A/B playback are planned.
- A failed capture can stop the current matching request. Large cold-cache searches have no background indexing progress or cancellation.
- Experimental phrase/coloring helpers exist, but a learned or source-invariant matcher is not integrated.

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
        │ one stdin/stdout JSON request per Python process
Audio package: services/audio/nam_tone_match
        ├── decode → slice → Demucs guitar stem
        ├── NAM render → spectral profile → ranking
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
| `POST` | `/api/tracks/:id/isolate` | `{startSec, endSec}` → selected-region metadata. |
| `GET` | `/api/tracks/:id/guitar` | Isolated guitar as WAV. |
| `POST` | `/api/tracks/:id/match` | `{results:[{name, distance, inputGainDb}]}`. |
| `GET` | `/api/tracks/:id/rendered/:name` | Probe phrase through a matched capture as WAV. |

Missing/invalid uploads and ranges return `400`; matching before isolation returns `409`; missing Demucs returns `503`. Ranges must remain inside the source duration and are never silently clamped. Uploads are limited to 300 MiB.

The only environment setting is `PORT`, defaulting to `3210`. There is no `.env` loader. If you change the API port during frontend development, also update the proxy in `apps/web/vite.config.js`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| “Web UI is not built” | Run `npm.cmd --prefix apps/web ci` and `npm.cmd --prefix apps/web run build` from the repository root. |
| “Demucs is not installed” | Install requirements with the same `python` the API finds on `PATH`, then restart the API. |
| MP3 decode fails with `_cffi_backend` missing | Check `python -m pip check` and `python -c "import miniaudio"`; repair/install `cffi` and `miniaudio` in that environment. WAV decoding uses the standard library. |
| NAM import fails or a DLL is blocked | Check the environment's PyTorch/NAM dependencies. The loader has an inference-only import fallback, but environment and architecture compatibility still matter. |
| No results | Confirm there are `.nam` files directly in `models/`; collecting into `nams/` alone requires the mirror option above. |
| First match appears slow | Uncached captures are being rendered synchronously. Start with a few captures to verify setup; inspect API errors if a capture fails. |
| Results remain stale after changing the renderer/probe | Remove the affected files in `data/profile-cache/` and rerun matching. |

Keep audio, generated artifacts, capture weights, and credentials out of commits. Source URLs and collection scripts are included so local libraries can be rebuilt while retaining their creators' terms.
