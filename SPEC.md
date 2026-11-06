# NAM Tone Match

Build a local web app that takes an MP3 or WAV, lets a person select a time range, isolates the guitar from that range, and ranks Neural Amp Modeler (NAM) captures by how close they sound to that guitar.

The person picks a region in the UI. The backend cuts that region out, separates the guitar stem, then runs one clean direct-input (DI) recording through every `.nam` file and compares each result to the isolated guitar. The closest capture wins.

Unit tests are the completion gate. Do not mark the work finished until `scripts/verify.ps1` prints `VERIFY_OK` and exits 0.

## Stack

- `apps/web` — React UI built with Vite. Plain JSX. No UI kit.
- `apps/api` — Node.js HTTP API. Use the built-in `node:http` and `node:test` modules. Do not add Express, a database, or a cloud host.
- `services/audio` — Python 3 package `nam_tone_match`, run with the `python` on PATH. NumPy is already installed. PyTorch is already installed.

Python calls cross the process boundary. The API writes a JSON request on stdin and reads one JSON object from stdout. The command is:

```text
python -m nam_tone_match
```

Run it with the working directory set to `services/audio`.

## What the UI does

One page:

1. Choose an MP3 or WAV file and upload it.
2. Draw the waveform from peak data the API returns.
3. Drag across the waveform to select a start and end, in seconds. Show the numeric range.
4. Button "Isolate guitar" sends that range and then plays the isolated guitar.
5. Button "Match NAM" ranks the captures and lists name, distance (lower is closer), and a control that plays the clean DI after that capture.

Keep the page usable with the keyboard: the file input and both buttons are real buttons or inputs with labels.

## HTTP API

Bind `127.0.0.1` and default the port to `3210`. JSON errors look like `{"error":"..."}` with a non-200 status.

| Method | Path | Body | Result |
| --- | --- | --- | --- |
| POST | `/api/tracks` | raw file bytes, header `x-filename` | `{id, name, durationSec, sampleRate}` |
| GET | `/api/tracks/:id/peaks` | | `{peaks:[number], durationSec}` about 1000 peaks in 0..1 |
| POST | `/api/tracks/:id/isolate` | `{startSec, endSec}` | `{id, startSec, endSec, durationSec}` |
| GET | `/api/tracks/:id/guitar` | | `audio/wav` of the isolated guitar |
| POST | `/api/tracks/:id/match` | | `{results:[{name, distance}]}` sorted by distance ascending |
| GET | `/api/tracks/:id/rendered/:name` | | `audio/wav` of the clean DI through that capture |

Store uploads under `data/` (gitignored) as `{id}.source` plus a `{id}.json` sidecar. Reject a missing file, a non-audio extension, and a range where `startSec` is not less than `endSec` or either end is outside the file. Clamp is not allowed. Return 400.

`:name` is the capture file name without directories. Reject `..`, slashes, and names that are not in the match result.

## Audio package

Modules under `services/audio/nam_tone_match/`:

- `decode.py` — load WAV with the standard library `wave` module, and MP3 with the `miniaudio` package (it bundles a decoder, so ffmpeg is not required). Return float32 mono samples in -1..1 and the sample rate. Downmix stereo by averaging.
- `slice.py` — cut a sample range from start/end seconds. This is pure NumPy.
- `peaks.py` — max absolute value per bucket.
- `isolate.py` — guitar separation.
- `match.py` — rank captures.
- `effects.py` — load effects.
- `cli.py` and `__main__.py` — stdin JSON commands `probe`, `peaks`, `isolate`, `match`, `render`.

`probe` returns duration and sample rate. `peaks` returns the peak array. `isolate` writes a WAV path. `match` returns the ranked list. `render` writes one capture's WAV.

### Guitar isolation

Production path: run Demucs model `htdemucs_6s` and keep the stem named `guitar`. Build the command in one function:

```python
def demucs_command(python_exe: str, input_wav: str, out_dir: str) -> list[str]:
    ...
```

The returned list must include `htdemucs_6s` and `guitar`. A second function takes a Demucs output directory and returns the path of the guitar stem file (the path contains the word `guitar`).

Do not download Demucs weights from a test. If the `demucs` module is missing, `isolate` still has a working fallback so the app can be tested: a mid/side-ish harmonic extractor is the wrong tool. Instead, when Demucs is missing, return exit status 0 with `{"error":"demucs_not_installed"}` from the CLI and let the API surface HTTP 503. Cover that branch with a unit test that stubs the importer.

Try `python -m pip install demucs miniaudio neural-amp-modeler` once. If a package fails to install, keep going. The verify script must still pass.

### NAM matching

A capture is anything in the `models/` directory ending in `.nam`.

Real NAM files are JSON. They contain `version`, `architecture`, `config`, and `weights`. When the `nam` package imports, load the file with `nam` and run the clean DI audio through the network. Resample the DI to the model's sample rate when the file has `sample_rate` (default 48000).

When a file is a test fixture instead of a neural net, it is JSON like `{"kind":"test-effect","type":"lowpass"}`. Supported types are `identity`, `lowpass`, and `gain`. Implement those with NumPy (a short moving average for lowpass, multiply by 0.5 for gain). This fixture exists so the ranker can be tested without a multi-megabyte amp capture.

Distance: mean squared error between log-magnitude STFT spectra of the isolated guitar and the rendered DI. Use NumPy `fft`. Same length: trim to the shorter one. Lower distance is a closer match. Sort ascending. Ties keep filename order.

The clean DI is generated, not recorded. `make_clean_di(sample_rate, seconds)` returns a few plucked notes: decaying sines at 196 Hz, 247 Hz, and 294 Hz, mixed together. The app writes `data/clean-di.wav` on demand at 48000 Hz, 2 seconds. Tests call `make_clean_di` themselves.

`rank_effects(target, clean, effects, sample_rate)` is the pure function under test. The closest of identity / lowpass / gain to a lowpassed copy of `clean` must be the lowpass effect.

## Tests

`scripts/verify.ps1` is the only completion command. It runs:

- `node --test` on `apps/api/test/*.test.js` and `apps/web/src/*.test.js`
- `python -m unittest discover -s services/audio -p "test_*.py"`

Required cases:

- WAV roundtrip: write a WAV, decode it, samples match within 1e-4.
- Slice rejects an inverted range and a range past the end.
- Peaks length is about 1000 and values are in 0..1.
- `demucs_command` contains `htdemucs_6s` and `guitar`.
- Stem picker returns the guitar file from a fake output tree.
- Lowpass fixture ranks ahead of identity and gain when the target is the lowpassed DI.
- A `.nam` missing `weights` raises a clear error.
- CLI `isolate` reports `demucs_not_installed` when the importer is stubbed.
- API: upload a WAV, fetch peaks, reject a bad range with 400, and with the Python CLI stubbed, isolate and match return the stubbed JSON.
- `clampRegion` in `apps/web/src/region.js` is a pure function. It does not clamp out-of-range values. It returns an error string when the range is inverted or outside the duration. Node can import it if it is plain JS, not JSX.

No browser test. No snapshot of a listening session.

## Layout

```text
apps/web/            Vite + React
apps/api/            Node API and its tests
services/audio/      Python package and its tests
models/README.md     how to drop .nam captures in models/
scripts/verify.ps1   completion gate
data/                runtime files, gitignored
```

## Rules for the agent

- Work only inside this directory.
- The shell is Windows PowerShell 5.1. It has no `&&`. Chain with `;` and check `$LASTEXITCODE`.
- Create files with real paths under this directory. A path that starts with `/` is wrong on this machine.
- Do not call `finish` until `powershell -NoProfile -File .\scripts\verify.ps1` prints a line that is exactly `VERIFY_OK` and exits 0.
- The finish message is one sentence that quotes `VERIFY_OK` and the test counts from that command's output.
- Do not invent files, passing tests, or captured audio that the tools did not produce.
- Update `README.md` so the quick start matches the commands that actually work.
