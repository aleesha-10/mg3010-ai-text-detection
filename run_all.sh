#!/usr/bin/env bash
# Usage: ./run_all.sh          (real data in data/raw/)
#        ./run_all.sh --demo   (synthetic smoke test; DO NOT report these results)
set -e
cd "$(dirname "$0")"
[ "$1" == "--demo" ] && python src/make_demo_data.py
python src/01_clean_data.py
python src/02_base_model.py
python src/03_extended_model.py
python src/04_robustness.py
python src/05_plots.py
