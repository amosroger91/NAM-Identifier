$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$failed = $false

$required = @(
  "apps\api\src\server.js",
  "apps\web\package.json",
  "services\audio\nam_tone_match\isolate.py",
  "services\audio\nam_tone_match\match.py",
  "services\audio\nam_tone_match\effects.py",
  "services\audio\nam_tone_match\cli.py",
  "services\audio\nam_tone_match\__main__.py"
)
foreach ($rel in $required) {
  if (-not (Test-Path -LiteralPath (Join-Path $root $rel))) {
    Write-Output "MISSING $rel"
    $failed = $true
  }
}

Write-Output "NODE_TEST_START"
node --test "apps/api/test/*.test.js" "apps/web/src/*.test.js"
if ($LASTEXITCODE -ne 0) { $failed = $true }

$apiTests = @(Get-ChildItem -LiteralPath (Join-Path $root "apps\api\test") -Filter *.test.js -ErrorAction SilentlyContinue)
if ($apiTests.Count -lt 1) {
  Write-Output "MISSING apps\api\test\*.test.js"
  $failed = $true
}

$packagePath = Join-Path $root "apps\web\package.json"
if (Test-Path -LiteralPath $packagePath) {
  $packageText = Get-Content -Raw -LiteralPath $packagePath
  if ($packageText -notmatch "react" -or $packageText -notmatch "vite") {
    Write-Output "MISSING react and vite in apps\web\package.json"
    $failed = $true
  }
}
$jsxFiles = @(Get-ChildItem -Recurse -LiteralPath (Join-Path $root "apps\web") -Filter *.jsx -ErrorAction SilentlyContinue)
if ($jsxFiles.Count -lt 1) {
  Write-Output "MISSING apps\web jsx page"
  $failed = $true
} else {
  $uiText = ($jsxFiles | ForEach-Object { Get-Content -Raw -LiteralPath $_.FullName }) -join "`n"
  if ($uiText -notmatch "Isolate guitar" -or $uiText -notmatch "Match NAM") {
    Write-Output "MISSING Isolate guitar and Match NAM controls"
    $failed = $true
  }
}
$serverPath = Join-Path $root "apps\api\src\server.js"
if (Test-Path -LiteralPath $serverPath) {
  $serverText = Get-Content -Raw -LiteralPath $serverPath
  if ($serverText -notmatch "/api/tracks" -or $serverText -notmatch "3210") {
    Write-Output "MISSING /api/tracks server on port 3210"
    $failed = $true
  }
}

Write-Output "PY_TEST_START"
python -m unittest discover -s services/audio -p "test_*.py"
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Output "BEHAVIOR_START"
python -c @"
import sys
sys.path.insert(0, r'services/audio')
import numpy as np
from nam_tone_match.isolate import demucs_command
cmd = demucs_command(sys.executable, 'in.wav', 'out')
assert isinstance(cmd, list) and 'htdemucs_6s' in cmd and any('guitar' in str(part) for part in cmd), cmd
from nam_tone_match.match import make_clean_di, rank_effects
clean = np.asarray(make_clean_di(48000, 2.0), dtype=np.float32)
kernel = np.ones(8, dtype=np.float32) / 8
target = np.convolve(clean, kernel, mode='same')
effects = [
    {'kind': 'test-effect', 'type': 'identity', 'name': 'identity.nam'},
    {'kind': 'test-effect', 'type': 'lowpass', 'name': 'lowpass.nam'},
    {'kind': 'test-effect', 'type': 'gain', 'name': 'gain.nam'},
]
ranked = rank_effects(target, clean, effects, 48000)
first = ranked[0]
name = first.get('name') if isinstance(first, dict) else str(first)
assert 'lowpass' in str(name), ranked
ranked_clean = rank_effects(clean, clean, effects, 48000)
clean_name = ranked_clean[0].get('name') if isinstance(ranked_clean[0], dict) else str(ranked_clean[0])
assert 'identity' in str(clean_name), ranked_clean
ranked_gain = rank_effects(clean * np.float32(0.5), clean, effects, 48000)
gain_name = ranked_gain[0].get('name') if isinstance(ranked_gain[0], dict) else str(ranked_gain[0])
assert 'gain' in str(gain_name), ranked_gain
print('BEHAVIOR_OK')
"@
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($failed) {
  Write-Output "VERIFY_FAILED"
  exit 1
}

Write-Output "VERIFY_OK"
exit 0
