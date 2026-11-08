# Plan for more accurate, realistic guitar tone matching

Date: 2026-11-08. Scope: improve the sound of the recommended, playable rig and establish that improvement on real guitar recordings. This document is a plan; the experiments and changes below have not been implemented or measured here.

## 1. Define the result we want

The app should return a small set of reproducible rigs that sound close to a selected reference: capture, input trim, cabinet treatment where needed, and any supported corrective processing. It should explain when the reference is too ambiguous or poorly isolated to support a useful recommendation.

The main problem is the unknown input. A recording contains the performance, guitar/pickups, pedals, amplifier, cabinet/microphone, and production processing. A dark recording can come from several different combinations. In the approximately linear case, the recorded spectrum contains the product of the source spectrum and the equipment responses; wet audio alone does not uniquely identify those factors. Better data and features can improve useful matching, but cannot guarantee recovery of the original equipment.

Support three clearly described cases:

| Case | Input available | Valid comparison |
| --- | --- | --- |
| Reference only, the default | Song or guitar stem | Compare tone across performances; return a diverse shortlist with uncertainty. |
| Reference plus the user's dry DI | Song/stem and the user's guitar take | Fit and audition a rig on that actual guitar. Different performances still require tone comparison. |
| Paired evaluation | Dry DI and its corresponding processed recording | Align the same performance and compare rendered audio directly. Use this to diagnose the system and evaluate known chains. |

A matching result should never imply that a digital input trim is the original amp's gain-knob setting.

## 2. Establish the actual starting point

Claude reports the following development results for 17 captures and 136 trials:

| Query source | Top-1 | Top-3 |
| --- | ---: | ---: |
| Synthetic phrases | About 57% | About 89% |
| Recorded guitar samples | About 35% | About 60% |

These are handoff results, not newly reproduced measurements. Targets were made by the captures themselves, without a full arrangement or separation, so they describe a relatively favorable setting. The broader-coloring/whitening experiment is unfinished; there is no established benefit to preserve yet.

The current code still uses a single synthetic probe, quarter-octave average spectra, four input trims, and a cached distance search. `phrases.py` and `signals.py` exist, but a trained multi-phrase matcher is not integrated into production.

The collection added in this chat has known provenance: `nams/catalog.jsonl`, `nams/sources/`, and `nams/LIBRARY_SUMMARY.json` record 1,602 distinct file hashes and their sources. The app's `models/` also contains the original captures. The imports passed structural loading and mirror checks. That does not establish full rendering compatibility, perceptual uniqueness, or retrieval accuracy. Keep the collection intact and use explicit manifests to select evaluation/indexing sets.

## 3. Milestone A — make comparisons reproducible

### Freeze the small benchmark

1. Recover and freeze Claude's original 17-capture set by content hash, filename, chosen submodel, and renderer version. The original 17 filenames are still present; confirm membership instead of benchmarking whatever happens to be in `models/`.
2. Preserve source WAV hashes, phrase definitions, gain values, dependency versions, and per-query rankings. Reproduce the unchanged baseline before tuning anything.
3. Treat those 136 cases as development/regression data: their failures have already guided decisions. Reserve new recording sessions for final evaluation.
4. Cross performance and input gain independently. The current script couples each seed to one gain, which prevents clean diagnosis of the two effects.
5. Audit the recorded sample library. Verify that its files are unprocessed electric-guitar DI, rather than already amplified, equalized, compressed, acoustic, or microphone recordings. Unknown source material can be a stress test, but cannot establish a dry-DI benchmark.

### Make the renderer trustworthy

- Preserve the legacy WaveNet loader fix and test representative old/new WaveNet, LSTM, and SlimmableContainer files. Compare rendered samples against a pinned official NAM engine on the same input and state. A valid weight count and finite output alone do not prove an equivalent config conversion.
- Test sample-rate conversion, output length, model reset/warm-up, and block continuity. Replace or correct the custom FFT resampler, including its Nyquist-bin edge behavior.
- Preserve input calibration and original DI level when known. Record any normalization or input trim explicitly; apply audition level matching after the nonlinear chain.
- Report and skip individual failed captures. Record searched/compatible/failed counts so a result cannot silently mean a much smaller search.

**Exit condition:** a replayable baseline report and a render-conformance report, with failures visible and no unexplained conversion or level changes. Do not begin a full-library benchmark at this stage.

## 4. Milestone B — replace the synthetic input assumption

Build a small, verified bank of recorded DIs covering single coils and humbuckers, clean picking and hard attacks, sustained notes, chords, palm muting, low/high registers, and at least some alternate tunings. Start with a manageable pilot of roughly 24–40 clips from several instrument/pickup/session groups; expand according to coverage and measured render cost.

Split original recordings into development, validation, and final test groups before cropping notes or making augmentations. All notes, crops, colorings, and level variants from one recording stay in its group. Hold out entire instruments/sessions where available, along with new performances. If only one guitar is available, explicitly limit the generalization claim.

Render each candidate on multiple development DIs and store a distribution or small set of prototypes. Retain spectral-envelope information, and compare the following experiments at a matched compute budget:

| Experiment | Change | What it must demonstrate |
| --- | --- | --- |
| B0 | Current single-probe score | Reproduced baseline. |
| B1 | Real-DI bank with the existing features | Benefit from fixing the input mismatch alone. |
| B2 | Several target windows; robust aggregation of prototype distances | Stability across notes and sections, without selecting one accidental best window. |
| B3 | Regularized feature weighting using within-capture variation across DIs, with cabinet state fixed and input level/trim controlled or stratified | Better discrimination on held-out guitars while retaining drive and amp/cabinet EQ information. |
| B4 | Additional harmonic, attack, sustain, and level-dependent spectral features | Improvement beyond B1–B3, including clean/crunch/high-gain breakdowns. |
| B5 | Simple regularized LDA, nearest-prototype, or kNN classifier | Whether a small learned model earns its added complexity. |

Use controlled DI colorings as augmentation, initially within plausible recorded-guitar ranges and applied before the amp. Compare real-only, synthetic-only, and mixed banks. Extreme random EQ is an experiment, not evidence that the source mismatch has been solved.

Two constraints matter:

- Do not independently flatten/whiten every target and candidate and assume the remaining signal describes the amp. That can remove the cabinet and amplifier EQ we need to match. Preserve an unwhitened spectral branch. Estimate nuisance weighting only on development data, preferably within a fixed capture and drive condition.
- Harmonic ratios and dynamics are additional clues, not input-independent fingerprints. Pickups, playing, chords, cabinet filtering, compression, and separation can change them. Use pitch-dependent features only in windows with reliable pitch estimates; keep a fallback for chords and distorted passages.

Fixed digital trim does not establish the same nonlinear excitation across DIs with different recorded levels and attacks. Preserve calibration where available, stratify those conditions, and explicitly check that nuisance weighting has not erased useful drive or cabinet distinctions.

**Advancement gate:** pre-register a meaningful improvement over the reproduced real-DI baseline, such as at least 10 percentage points in top-3, with a positive paired confidence interval and no unexplained material loss in a major tone group. This is a proposed decision threshold, not a performance forecast. Bootstrap by independent performance/session groups, not by treating gain variants as independent recordings.

## 5. Milestone C — match the audible signal chain

Create one canonical renderer used by fitting, preview, and export:

```text
dry DI → calibrated/adjustable input trim → NAM capture
       → optional cabinet IR → bounded post-EQ → audition output trim
```

1. Classify captures as amp head, amp+cab/full rig, pedal/preamp, power amp, or unknown. Preserve uncertainty and permit correction; filenames and metadata are sometimes inconsistent. Keep pedal and bass collections available, but route them to appropriate chains rather than treating every file as a standalone guitar amp.
2. Search compatible amp/cabinet combinations. Use a small, varied IR bank for amp-head captures. Do not add a second cabinet to an amp+cab capture by default.
3. Use the index to shortlist candidates, then refine input trim, cabinet choice, and a small number of bounded EQ controls. Penalize large corrections and extreme boosts. Track unfitted and fitted scores separately so flexible EQ cannot conceal poor capture selection.
4. Honor NAM input/output calibration metadata when present. Calibration documents the digital/analog level relationship; missing calibration remains unknown. It cannot recover the original song's guitar output level. [NAM calibration documentation](https://github.com/sdatkinson/neural-amp-modeler/blob/main/docs/source/tutorials/calibration.rst)
5. Separate `probeInputTrimDb`, actual own-DI trim, and output playback trim. Present fitted trim as a setting for the selected input, never as the source amp's original setting.
6. Add reverb/delay/compression handling only after base amp/cab matching improves. Report residual production differences. When supported, evaluate ambience separately from dry timbre so extra reverb cannot win the tone comparison by itself.

Export the selected capture/IR identifiers and parameters as a preset plus rendered audio. An IR/EQ-assisted chain does not automatically become a single equivalent `.nam` file.

**Exit condition:** the fitted chain improves independent listening judgments as well as the optimization score on paired material, and improves useful matching on held-out unpaired guitars/performances. Paired fitting alone cannot validate the default reference-only task. Preview and export must reproduce the exact chain that was evaluated.

## 6. Milestone D — preserve and evaluate real song information

- Preserve original stereo through decoding, region preparation, Demucs, and saved stems. Derive mono explicitly for a particular feature extractor; offer left/right analysis for panned guitars. The current code averages stereo before separation and then duplicates that mono signal.
- Separate a region with surrounding context, then trim back to the selected time range. Evaluate the padding amount instead of assuming a fixed crop is optimal.
- Keep processing in float arrays. If float WAV intermediates are introduced, add float-WAV decoding in the same change; the current standard-library WAV path cannot simply be assumed to support them. Avoid accidental clipping and unnecessary repeated 16-bit conversion.
- Keep source/stem playback and region looping available. Reject silence, very short usable content, and clearly unsuitable inputs. Estimate stem reliability conservatively; a separator output label is not proof of a clean guitar.
- Score several useful windows and expose disagreement. A passage containing different guitars or changing tones may require separate matches.
- Allow direct isolated-stem upload, bypassing separation when better source material exists.
- Benchmark alternative guitar separators only if separation is a measured bottleneck. Pin versions and compare end-to-end matching and listening, in addition to separation metrics. Existing Demucs documentation confirms the guitar-capable six-source model and configurable processing options. [Demucs documentation](https://github.com/facebookresearch/demucs)

Use a progressive evaluation ladder:

| Tier | Reference/query construction | Purpose |
| --- | --- | --- |
| Renderer control | Same known DI, capture, and gain | Detect renderer, alignment, or indexing errors. |
| Different real DI | Held-out performances through known captures | Test source/instrument robustness. |
| Produced guitar | Known rig plus controlled EQ, compression, ambience, or codec processing | Measure production sensitivity. |
| Mixed song | Known guitar added to an arrangement and then separated | Compare known guitar stem with separated output to isolate separation damage. |
| Open set | True family omitted; silence, other instruments, heavy bleed, multiple guitars | Test family detection, useful alternatives, and uncertainty separately. |
| Real reference song | Representative song passages | Judge practical similarity by listening; do not invent equipment labels. |

Distinguish a known isolated guitar stem from the unavailable dry DI. Having the former does not remove the unknown-performance problem.

An omitted original family does not prove that no convincing match exists: a different family may sound close. Keep missing-family labels separate from perceptual suitability, and obtain listening/relevance labels for decisions that say no usable chain is available.

## 7. Milestone E — use the large library effectively

Keep all downloaded files, but separate file identity, render equivalence, amp family, gain/cabinet variants, and perceptual similarity.

- Add normalized network fingerprints that ignore display metadata and compare effective config, float32 weights, sample rate, and selected submodel. Verify equivalence conservatively. Byte-level deduplication has already happened; tonal equivalence has not.
- Preserve near neighbors as hard cases. A low top-1 result is not automatically wrong when settings sound similar, but do not relabel every confusing pair as equivalent after seeing the errors.
- Begin broader evaluation with roughly 60–100 compatible captures balanced across families and tone types. The existing library contains many variants of a few amps; raw capture count must not dominate the metrics.
- Index family representatives first, then their settings. Return several distinct useful alternatives, with nearby variants expandable beneath them. Measure shortlist recall so representative selection does not hide the best match.
- Use an explicit, resumable indexing job with bounded memory and concurrency. Search ready entries immediately and show coverage. Avoid a full render sweep on every request or a mandatory full-library build at server start.
- Key cached artifacts by capture content hash, renderer/version, submodel, DI hash, input trim, IR/chain assumptions, feature version, and training split. Keep training/validation/test artifacts separate. Measure rendering on a representative small batch before choosing the full indexing budget.

Report strict capture top-1/3/5 and MRR, family top-k, distinct shortlist families, compatibility coverage, and latency. Macro-average across families and show separate results for clean/crunch/high-gain, cabinet states, and instrument/session groups. Calibrate original-family detection on known/absent-family cases; calibrate “no convincing match” on independently judged usable/unusable results, including alternatives from other families. Winner/runner-up distance is not a probability.

**Exit condition:** broad-library results remain useful across families and realism tiers; the expanded candidate count does not merely produce a more confident-looking arbitrary winner.

## 8. Milestone F — make listening the release gate

Add one exclusive A/B transport with looping and level matching after the complete chain. Support the user's actual DI as audition material. In paired tests, hold the dry performance constant across reference, old selection, and new selection. For commercial songs with no paired DI, clearly label different-performance comparisons.

Run randomized, hidden-label comparisons with several listeners, a known-good reference and poor anchor, and repeated items for consistency. Ask both “which is closer?” and “is at least one shortlist result usable?” Compare uncertainty across listeners and independent reference clips. Do not count repeated button presses as independent evidence.

Also fix reference identity: changing the selected range invalidates the previous stem, matches, and audition URLs. Give each isolation/result an immutable revision so old responses cannot restore the wrong audio. This is required for meaningful listening judgments.

A release needs reproducible improvement on untouched recorded DIs, produced/separated material, and blinded listening. Retain strict retrieval scores even when family or preference scores improve. Set final numerical release targets after the reproduced baseline and pilot data establish the achievable task and statistical uncertainty.

## 9. Learned representations: a later, bounded experiment

If the real-DI bank and simple classifiers plateau, train a guitar-specific similarity encoder on rendered data. Pair different performances processed through the same defined tone condition, use close gain/cabinet variants as deliberate negatives where appropriate, balance families, and hold out instruments and sessions. Keep source variation separate from the cabinet/processing attributes the model is supposed to recognize.

Prototype on the small frozen library first. Compare with B1–B5 at the same candidate coverage and evaluate both known-library retrieval and absent-family behavior. A per-library classifier is a closed-set solution unless rejection is separately demonstrated.

Relevant research provides directions, not a ready solution:

- ST-ITO studies learned production-style similarity and parameter search, but its authors explicitly report difficulty with guitar tone matching. Do not adopt its embedding as a presumed fix. [ST-ITO, discussion](https://arxiv.org/html/2410.21233v1#S6)
- Guitar-specific contrastive tone embeddings are a relevant design reference; verify access to models/data and applicable terms before planning around a pretrained artifact. [Towards zero-shot amplifier modeling](https://arxiv.org/abs/2407.10646)
- Tone removal/DI reconstruction is an optional research path. Any recovered DI is an estimate that can change the performance; test it on known paired material before using it to drive candidate matching. [EG-VAE](https://arxiv.org/abs/2608.05513)

Stop or redirect an experiment when it only improves synthetic cases, wins on its own score without listening benefit, requires test-set tuning, or loses important clean/drive/cabinet distinctions. Acquire better real DI data or improve the own-DI workflow instead of indefinitely adding synthetic colorings.

## 10. Implementation order and file boundaries

| Order | Deliverable | Main files |
| --- | --- | --- |
| 1 | Frozen 17-capture baseline, grouped DI manifest, renderer checks | `scripts/accuracy_check.py`, `effects.py`, proposed benchmark manifests/reports |
| 2 | Real-DI bank and B0–B5 comparison with ablations | `match.py`, `phrases.py`, proposed feature/index modules |
| 3 | Canonical amp/cabinet chain and bounded refinement | `effects.py`, `match.py`, proposed `chain.py`, `cli.py` |
| 4 | Stereo/context preservation and progressive mix evaluation | `decode.py`, `encode.py`, `isolate.py`, benchmark harness |
| 5 | Immutable region/result identity, own-DI audition and A/B playback | `apps/api/src/server.js`, `apps/web/src/App.jsx`, `cli.py` |
| 6 | Balanced broader library, resumable indexing, calibrated coverage/rejection | index/catalog modules, API status, UI results |
| 7 | Optional guitar-specific embedding experiment | separate experiment/training module; promote only after the same gates |

Reference revision fixes can proceed alongside orders 1–2. Keep the local React/Node/Python structure. Update the scoring claims in `SPEC.md` and `README.md` when a new production method is adopted. Add tests for the actual production ranker and chain behavior; retain existing fixture tests as plumbing checks. Continue requiring `scripts/verify.ps1` for implementation changes, alongside the independent audio-quality reports.

The first implementation batch should freeze and reproduce the 17-capture baseline, establish the real-DI split, validate the legacy renderer, and compare a real-DI bank against the current score. That experiment decides the next matching method before spending the full-library indexing budget.
