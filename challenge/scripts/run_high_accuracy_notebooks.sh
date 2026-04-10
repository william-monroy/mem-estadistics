#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_01_KNN_Tuned.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_04_RandomForest_Tuned.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_05_GradientBoosting_Tuned.ipynb
