param(
    [ValidateSet("check", "unit", "synthetic", "smoke")]
    [string]$Mode = "unit"
)

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
} else {
    $Python = "python"
}

& $Python (Join-Path $PSScriptRoot "run_test_pack.py") --mode $Mode
exit $LASTEXITCODE
