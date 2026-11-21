#!/usr/bin/env bash
# Usage: ./run_all.sh          (real data in data/raw/)
#        ./run_all.sh --demo   (synthetic smoke test; DO NOT report these results)
# Once before the first run:
#   pip install -r requirements.txt
#   python src/10_make_emotion_lexicon.py      # writes data/raw/emotion_lexicon.csv (needs nrclex)
# Fresh clone without data/raw/: python src/00_rebuild_raw_from_clean.py
set -e
cd "$(dirname "$0")"
[ "$1" == "--demo" ] && python src/make_demo_data.py
python src/01_clean_data.py                  # cleaning + grouped train/val/test split
python src/02_base_model.py                  # base model
python src/03_extended_model.py              # extended model, bootstrap CIs, McNemar
python src/04_robustness.py                  # perturbations, unseen generators
python src/08_top_features.py                # top features (explainability)
python src/09_repeated_splits.py             # 5 grouped splits, nested models, transfer
python src/11_artefact_checks.py             # shortcut / artefact checks
python src/12_calibration_and_cleaning.py    # exploratory: calibrated threshold + cleaned corpus
python src/05_plots.py                       # figures (400 dpi PNG + PDF)
