# Windows PowerShell equivalent of run_all.sh. Run from the project folder with the venv active:
#   .\run_all.ps1          (real data in data\raw\)
#   .\run_all.ps1 -Demo    (synthetic smoke test; DO NOT report these results)
# Once before the first run:
#   pip install -r requirements.txt
#   python src/10_make_emotion_lexicon.py      # writes data/raw/emotion_lexicon.csv (needs nrclex)
# Fresh clone without data/raw/: python src/00_rebuild_raw_from_clean.py
param([switch]$Demo)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($Demo) { python src/make_demo_data.py; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
$steps = "01_clean_data.py", "02_base_model.py", "03_extended_model.py", "04_robustness.py", "08_top_features.py",
         "09_repeated_splits.py", "11_artefact_checks.py", "12_calibration_and_cleaning.py", "05_plots.py"
foreach ($s in $steps) {
    Write-Host "== $s"
    python "src/$s"
    if ($LASTEXITCODE) { throw "src/$s failed (exit code $LASTEXITCODE)" }   # native exit codes do not trigger Stop
}
