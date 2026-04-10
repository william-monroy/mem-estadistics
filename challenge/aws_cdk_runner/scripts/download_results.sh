#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/deploy_config.json}"
DEST_DIR="${2:-$ROOT_DIR/downloads}"

eval "$(python3 "$ROOT_DIR/scripts/emit_deploy_env.py" --config "$CONFIG_PATH")"

if [[ -n "${ULTRA_CDK_AWS_PROFILE:-}" ]]; then
  export AWS_PROFILE="$ULTRA_CDK_AWS_PROFILE"
fi

if [[ -n "${ULTRA_CDK_AWS_REGION:-}" ]]; then
  export AWS_REGION="$ULTRA_CDK_AWS_REGION"
  export AWS_DEFAULT_REGION="$ULTRA_CDK_AWS_REGION"
fi

RESULTS_BUCKET="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ResultsBucketName'].OutputValue" --output text)"
RESULTS_PREFIX="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ResultsPrefix'].OutputValue" --output text)"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TARGET_DIR="$DEST_DIR/$ULTRA_CDK_STACK_NAME-$TIMESTAMP"
mkdir -p "$TARGET_DIR"

aws s3 sync "s3://$RESULTS_BUCKET/$RESULTS_PREFIX/" "$TARGET_DIR"

echo "Results downloaded to: $TARGET_DIR"
