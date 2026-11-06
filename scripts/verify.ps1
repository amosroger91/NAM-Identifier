$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$failed = $false

Write-Output "NODE_TEST_START"
node --test "apps/api/test" "apps/web/src"
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Output "PY_TEST_START"
python -m unittest discover -s services/audio -p "test_*.py"
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($failed) {
  Write-Output "VERIFY_FAILED"
  exit 1
}

Write-Output "VERIFY_OK"
exit 0
