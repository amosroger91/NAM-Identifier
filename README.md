# NAM Tone Match

Local web app that isolates a guitar from a chosen slice of a song and ranks Neural Amp Modeler captures against that tone.

## Overview

A mixed MP3 hides the guitar inside the rest of the arrangement. This app cuts out the section you select, separates the guitar stem, and compares that stem with a clean direct-input recording run through each NAM capture in `models/`. The closest capture is the one whose rendered DI has the smallest spectral distance to the isolated guitar.

The build contract is `SPEC.md`. Unit tests are the gate. `scripts/verify.ps1` prints `VERIFY_OK` when they pass.

## Architecture

```text
browser (Vite React)
    |  HTTP 127.0.0.1:3210
apps/api  (Node)
    |  stdin/stdout JSON
services/audio  (Python: decode, slice, Demucs guitar stem, NAM render, STFT distance)
    |
models/*.nam
```

Demucs model `htdemucs_6s` supplies the guitar stem when that package is installed. NAM files are rendered with the `nam` package when it is installed. Test fixtures use small NumPy effects so the ranker can be proven without a commercial amp capture.

## Quick start

From this directory, after the implementation exists:

```powershell
python -m pip install miniaudio
npm install
npm --prefix apps/web install
npm --prefix apps/web run build
node apps/api/src/server.js
```

Open `http://127.0.0.1:3210/`. Put `.nam` files in `models/`. The app generates `data/clean-di.wav` itself.

Run the gate:

```powershell
powershell -NoProfile -File .\scripts\verify.ps1
```

## Configuration

No `.env` file. Paths are local.

| Name | Required | Description |
| --- | --- | --- |
| `PORT` | optional | API port. Default `3210`. |
| `models/` | optional | Directory of `.nam` captures. Empty until you add files. Test fixtures cover the ranker. |

## Usage / HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/tracks` | Upload raw audio. Header `x-filename`. |
| GET | `/api/tracks/:id/peaks` | Waveform peaks. |
| POST | `/api/tracks/:id/isolate` | Body `{startSec, endSec}`. Guitar stem for that range. |
| GET | `/api/tracks/:id/guitar` | WAV of the isolated guitar. |
| POST | `/api/tracks/:id/match` | Rank NAM captures. Lower distance is closer. |
| GET | `/api/tracks/:id/rendered/:name` | WAV of the clean DI through that capture. |

## Project structure

```text
SPEC.md                 build contract
apps/web                React UI
apps/api                Node API and tests
services/audio          Python audio package and tests
models/README.md        where captures go
scripts/verify.ps1      unit-test gate
```

## Tech stack

- React and Vite
- Node.js `node:http` and `node:test`
- Python, NumPy, PyTorch
- Demucs `htdemucs_6s` for the guitar stem
- Neural Amp Modeler (`.nam` JSON) for amp and pedal captures
- miniaudio for MP3 decode

## Notes / conventions

Secrets do not belong in this repo. Audio uploads and generated WAV files stay in `data/`. Real `.nam` captures stay in `models/` and are gitignored because they are large community files, not source. The checked-in ranker fixtures are small JSON effects, not amp models.
