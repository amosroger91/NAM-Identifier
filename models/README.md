# NAM captures

Put Neural Amp Modeler files here. Each file ends in `.nam` and is the JSON capture of an amp or pedal.

The matcher runs `data/clean-di.wav` through every file in this directory and ranks them against the isolated guitar. A lower distance is a closer tone.

These files are gitignored. Download captures from the community (Tone3000 and the NAM Hub publish them) and drop them in this folder. The unit tests do not need a real capture. They use the small `test-effect` JSON fixtures described in `SPEC.md`.
