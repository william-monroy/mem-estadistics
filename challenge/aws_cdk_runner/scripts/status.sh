#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/deploy_config.json}"

eval "$(python3 "$ROOT_DIR/scripts/emit_deploy_env.py" --config "$CONFIG_PATH")"

if [[ -n "${ULTRA_CDK_AWS_PROFILE:-}" ]]; then
  export AWS_PROFILE="$ULTRA_CDK_AWS_PROFILE"
fi

if [[ -n "${ULTRA_CDK_AWS_REGION:-}" ]]; then
  export AWS_REGION="$ULTRA_CDK_AWS_REGION"
  export AWS_DEFAULT_REGION="$ULTRA_CDK_AWS_REGION"
fi

STACK_STATUS="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query 'Stacks[0].StackStatus' --output text)"
echo "Stack status: $STACK_STATUS"
echo
aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query 'Stacks[0].Outputs' --output table
echo

INSTANCE_ID="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)"
RESULTS_BUCKET="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ResultsBucketName'].OutputValue" --output text)"
RESULTS_PREFIX="$(aws cloudformation describe-stacks --stack-name "$ULTRA_CDK_STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='ResultsPrefix'].OutputValue" --output text)"

if [[ -n "$INSTANCE_ID" && "$INSTANCE_ID" != "None" ]]; then
  aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,LaunchTime:LaunchTime,PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress}' \
    --output table
  echo
fi

if [[ -n "$RESULTS_BUCKET" && -n "$RESULTS_PREFIX" ]]; then
  echo "Latest objects in s3://$RESULTS_BUCKET/$RESULTS_PREFIX/"
  aws s3 ls "s3://$RESULTS_BUCKET/$RESULTS_PREFIX/" --recursive | tail -n 20 || true
fi
