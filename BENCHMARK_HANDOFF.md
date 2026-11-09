# Implement the accuracy measurement

Measure how much the recorded-plus-synthetic prototype bank improves capture retrieval over the original single-probe matcher. Use the frozen 17-capture set and publish a reproducible comparison. The primary measure is the change in exact-capture top-3 retrieval. Keep this work focused on the benchmark harness and report.

## Start from the current working tree

The earlier handoff predates additional work. Inspect `git status` and preserve the uncommitted application changes. `services/audio/nam_tone_match/bank.py` and `match.py` already implement nearest-prototype matching with recorded and synthetic phrases and a six-trim production index. Capture failure reporting, the `probeInputTrimDb` field, and isolation revisions are present in the working tree. Use the single-probe B0 scorer in `scripts/run_bench.py` as the historical baseline.

The current README records a preliminary comparison on the black guitar. Treat it as a previous experiment using the original split and unequal gain grids.

| Previous experiment, 510 trials | Top-1 | Top-3 | MRR |
| --- | ---: | ---: | ---: |
| Single synthetic probe | 23.5% | 40.6% | 0.387 |
| Recorded + synthetic bank, nearest prototype | 39.6% | 60.4% | 0.544 |

The README reports a +19.8-point top-3 change with a paired interval of +14.9 to +24.7 points. `data/e2e-mix.log` also records a seven-mix smoke test of the bank at 43% top-1 and 71% top-3. The mix run is too small and lacks a paired baseline. Keep both results labeled as prior evidence. The black guitar has already been evaluated, so the corrected run is a replication on that instrument.

At handoff, `data/bench-cache/` contained 3,231 `.npz` files from the original run. That count includes at least one entry outside the 3,230 expected combinations. `scripts/bench_protocol.py` exists and is untracked. `benchmarks/measurement_protocol.json` has yet to be generated. Inspect the cache and running processes before starting renders. Two overlapping worker batches were previously found and stopped. Use one coordinated run with disjoint capture partitions and explicit worker exit checks.

## 1. Freeze the corrected experiment

1. Use `benchmarks/baseline17.json` as the candidate list. Verify the exact filenames, SHA256 hashes, and byte sizes. Include all 17 captures, including the pedal and neighboring amp settings. Missing or invalid inputs should fail the benchmark with a clear diagnostic.
2. Use `scripts/bench_protocol.py` to freeze and load `benchmarks/measurement_protocol.json`. `freeze_protocol(samples_root, models_root=...)` records the phrase definitions. `load_protocol(..., samples_root=..., models_root=...)` checks their integrity and source files. Supply the sample root through `--samples` or an environment variable. The existing library is under `Music/Sample Libraries/Black and Green Guitars/Samples`. Persist relative paths and hashes only.
3. Materialize recorded phrases from each manifest entry's ordered `notes`, `offset_samples`, and `output_samples`. Preserve the existing decoding, resampling, note fade, and peak scaling. Preserve the eight green bank phrases. Use the protocol's ten new green query phrases and ten new black query phrases. They exclude shared source WAV hashes across the bank and queries and among the queries themselves.
4. Use green and synthetic queries for development. Keep black out of development scoring and fitting. The library is made of assembled notes from two instruments with an unverified recording signal path. Describe the final comparison as a cross-instrument stress test.
5. Give B0 and all candidate methods the same four bank trims, `−6, 0, +6, +12 dB`. Cross every query independently with `−6, 0, +6 dB`. This controls for the extra gain choices the earlier B1 comparison had. Label any separate six-trim measurement as a production-configuration comparison.

## 2. Render and cache the inputs

- Wire `scripts/run_bench.py` and `scripts/bench_lib.py` to the frozen protocol. Preserve their existing entry points where practical. Provide commands to freeze the protocol, fill the render cache, compare on development data, and evaluate the frozen selection. Document the actual commands in the report.
- Key the corrected cache by capture SHA256, exact input samples or their hash, sample rate, trim, output normalization, renderer identity, and feature implementation/version. Record runtime versions in the run report. Precompute each phrase once per worker and reuse each loaded capture across its phrases and trims.
- Preserve the original cache for reproducing the earlier experiment. The old `r1` key includes a phrase label but omits its source audio hash and feature identity. Use a new namespace for the corrected run. Adopt an older entry only where its inputs and calculation can be established as identical. Record any reuse in the report. The revised green and black query inputs require new features.
- Write features atomically and rebuild unreadable or invalid entries. Validate the expected combinations and finite features before scoring. Refuse partial coverage for the primary comparison. For eight green bank phrases, eight synthetic bank phrases, one probe, and three ten-phrase query sets, the common-trim grid is `17 × [(1 + 8 + 8) × 4 + (10 + 10 + 10) × 3] = 2,686` entries.
- Use bounded worker counts and one thread budget per worker. `data/bench-profile.json` contains timings on two captures. One to three Torch threads had similar performance, with the faster setting depending on capture. Measure sustained throughput of the coordinated job. Record completed entries and report failures rather than relying on `ALL_DONE` from the old shell launcher.

## 3. Select on development data, then evaluate black

1. Compare B0, the recorded-bank and mixed-bank prototypes, multi-window scoring, weighted features, and dynamics features on the same green and synthetic trials. Keep the existing method definitions unless a correctness fix is necessary. For multi-window scoring, evaluate each trim across the windows before minimizing over trims so all windows share one trim. Reject nonfinite scores and preserve the frozen capture order for ties.
2. Select the challenger by green top-3, with green MRR and then the simpler method as tie-breakers. Synthetic results are diagnostic. Include the shipped nearest-prototype scoring approach in the comparison. For the controlled experiment, constrain it to the common four trims.
3. Save the selection before reading black results. Include the protocol ID, candidate and code fingerprints, method parameters and fitted weights where applicable, development metrics, gain grid, and gate. Final evaluation should read this selection rather than accepting an arbitrary method argument. Allow exact reruns of the same frozen comparison.
4. Run B0 and the selected method on all 510 black trials. Save per-trial ranks and capture scores with the query ID, capture hash, and input trim. Report top-1, top-3, MRR, and per-capture results. Show where the proposed method loses accuracy, especially on clean captures and adjacent settings.
5. Compute a paired 95% interval for the top-3 change by resampling the ten complete composite query groups with replacement. Keep both methods, all captures, and all trims together within a group. Record the seed and number of draws. The interval describes variation across assembled phrases within this one library and instrument. It cannot estimate variation across independent guitars, players, or sessions.

**Decision rule.** The measured stress-test gate is met if top-3 improves by at least 10 percentage points and the paired 95% interval has a lower bound above zero. Report the result even if the gate fails. A change in this benchmark supports a claim about the frozen capture set and sampled-note library. A real-song accuracy claim requires independent recordings and paired mixture/separation tests.

## 4. Deliver the evidence

- Write a short Markdown results report under `benchmarks/` with the controlled result, interval, gate decision, development selection, coverage, and limitations. Include exact commands to reproduce it. Link the report from the README and correct the claim of ten independent performances.
- Save the protocol, frozen selection, metrics, and per-trial results. Keep source audio, weights, render caches, and machine-specific paths in local ignored storage or outside the repo.
- Check source disjointness, cache invalidation, stable ties, and separation of development from black scoring. Include one B0 parity check against the historical profile scorer on a fixture. Run `scripts/verify.ps1` and report its actual result. The unit gate and the measured accuracy gate answer different questions.
- Keep the active production ranker out of this experiment's decision path. If the measured winner differs from the shipped method, document the difference for a separate implementation decision.
