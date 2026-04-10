#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_01_KNN_Ready.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_02_LogisticRegression_Ready.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_03_DecisionTree_Ready.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_04_RandomForest_Ready.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=0 Challenge_05_GradientBoosting_Ready.ipynb
