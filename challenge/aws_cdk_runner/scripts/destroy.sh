#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/deploy_config.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Deploy config not found: $CONFIG_PATH" >&2
  exit 1
fi

"$ROOT_DIR/scripts/setup_local.sh"
eval "$("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/emit_deploy_env.py" --config "$CONFIG_PATH")"

if [[ -n "${ULTRA_CDK_AWS_PROFILE:-}" ]]; then
  export AWS_PROFILE="$ULTRA_CDK_AWS_PROFILE"
fi

if [[ -n "${ULTRA_CDK_AWS_REGION:-}" ]]; then
  export AWS_REGION="$ULTRA_CDK_AWS_REGION"
  export AWS_DEFAULT_REGION="$ULTRA_CDK_AWS_REGION"
fi

export ULTRA_RUNNER_DEPLOY_CONFIG="$ULTRA_CDK_CONFIG_PATH"

(
  cd "$ROOT_DIR"
  npx aws-cdk destroy "$ULTRA_CDK_STACK_NAME" -f
)
