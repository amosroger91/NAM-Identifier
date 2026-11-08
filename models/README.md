# NAM captures

Put Neural Amp Modeler files here. Each file ends in `.nam` and is the JSON capture of an amp or pedal.

The matcher runs the 12-second synthetic phrase in `data/probe-di.wav` through every file directly in this directory, tests four input trims, and compares level-normalized spectral profiles with the isolated guitar. A lower distance means a closer match under that experimental score.

Capture files are gitignored and are not supplied by cloning the repository. Add your own captures or use `scripts/collect_nams.py` with `--mirror-dir models`; see the repository README for complete commands, source manifests, and provenance details. The app does not scan nested directories.

Keep each creator's terms and attribution with the files. Structure checks do not establish rendering compatibility or audio quality. The unit tests use small `test-effect` JSON fixtures and do not need real captures.
