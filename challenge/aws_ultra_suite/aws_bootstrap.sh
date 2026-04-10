#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SUITE_DIR/config.example.json}"
PROFILE_OVERRIDE="${2:-}"
VENV_DIR="$SUITE_DIR/.venv"
CACHE_DIR="$SUITE_DIR/.cache"

echo "[aws-ultra] suite dir: $SUITE_DIR"
echo "[aws-ultra] config: $CONFIG_PATH"

mkdir -p "$CACHE_DIR/matplotlib" "$CACHE_DIR/fontconfig"
export XDG_CACHE_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$SUITE_DIR/requirements.txt"

RUN_CMD=(python "$SUITE_DIR/run_all_ultra_models.py" --config "$CONFIG_PATH")
if [[ -n "$PROFILE_OVERRIDE" ]]; then
  RUN_CMD+=(--profile "$PROFILE_OVERRIDE")
fi

echo "[aws-ultra] starting run: ${RUN_CMD[*]}"
PYTHONUNBUFFERED=1 "${RUN_CMD[@]}"
