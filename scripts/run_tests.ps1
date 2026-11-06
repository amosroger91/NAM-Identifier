#!/usr/bin/env powershell

# Run all tests in the tests directory

$testPath = "Z:/profiles/roger/Documents/dev/nam-tone-match/tests"

# Change to the test directory
Set-Location $testPath

# Run Python unittest discovery
python -m unittest discover .

# Return to original directory
Set-Location "Z:/profiles/roger/Documents/dev/nam-tone-match"
