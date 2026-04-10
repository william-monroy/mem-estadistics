from __future__ import annotations

import json
import textwrap

from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_assets as s3_assets

from .config import DeployConfig


RUN_ROOT = "/opt/ultra-runner"


def build_runtime_config_payload(deploy_config: DeployConfig, *, results_s3_uri: str) -> dict:
    return {
        "workspace_root": f"{RUN_ROOT}/workspace",
        "data_dir": f"{RUN_ROOT}/data",
        "output_root": f"{RUN_ROOT}/workspace/output",
        "submissions_root": f"{RUN_ROOT}/workspace/submissions",
        "exports_root": f"{RUN_ROOT}/workspace/exports",
        "logs_root": f"{RUN_ROOT}/workspace/logs",
        "temp_root": f"{RUN_ROOT}/workspace/tmp",
        "profile": deploy_config.profile,
        "models": deploy_config.models,
        "continue_on_error": deploy_config.continue_on_error,
        "reserve_cpus": deploy_config.reserve_cpus,
        "max_cpu_fraction": deploy_config.max_cpu_fraction,
        "svm_parallel_jobs": deploy_config.svm_parallel_jobs,
        "svm_cache_mb": deploy_config.svm_cache_mb,
        "knn_n_jobs": deploy_config.knn_n_jobs,
        "ensemble_n_jobs": deploy_config.ensemble_n_jobs,
        "s3_data_uri": deploy_config.s3_data_uri,
        "s3_output_uri": results_s3_uri,
        "sync_outputs_every_stage": deploy_config.sync_outputs_every_stage,
        "create_resume_bundles": deploy_config.create_resume_bundles,
        "include_data_in_bundles": deploy_config.include_data_in_bundles,
    }


def build_user_data_script(
    deploy_config: DeployConfig,
    *,
    suite_asset: s3_assets.Asset,
    data_asset: s3_assets.Asset | None,
    results_bucket: s3.Bucket,
    results_prefix: str,
    region: str,
) -> str:
    results_s3_uri = f"s3://{results_bucket.bucket_name}/{results_prefix}"
    suite_asset_uri = f"s3://{suite_asset.s3_bucket_name}/{suite_asset.s3_object_key}"
    data_asset_uri = f"s3://{data_asset.s3_bucket_name}/{data_asset.s3_object_key}" if data_asset else ""
    runtime_payload = build_runtime_config_payload(deploy_config, results_s3_uri=results_s3_uri)
    runtime_json = json.dumps(runtime_payload, indent=2, sort_keys=True)

    return textwrap.dedent(
        f"""
        set -euxo pipefail
        exec > >(tee /var/log/ultra-runner-user-data.log | logger -t ultra-runner-user-data -s 2>/dev/console) 2>&1

        RUN_ROOT={json.dumps(RUN_ROOT)}
        SUITE_DIR="$RUN_ROOT/aws_ultra_suite"
        DATA_DIR="$RUN_ROOT/data"
        WORKSPACE_DIR="$RUN_ROOT/workspace"
        SUITE_ASSET_URI={json.dumps(suite_asset_uri)}
        DATA_ASSET_URI={json.dumps(data_asset_uri)}
        RESULTS_S3_URI={json.dumps(results_s3_uri)}
        AWS_REGION={json.dumps(region)}
        TERMINATE_ON_SUCCESS={json.dumps(str(deploy_config.terminate_on_success).lower())}
        TERMINATE_ON_FAILURE={json.dumps(str(deploy_config.terminate_on_failure).lower())}

        export AWS_REGION
        export AWS_DEFAULT_REGION="$AWS_REGION"

        command -v dnf >/dev/null 2>&1
        dnf install -y python3 python3-pip unzip tar gzip procps-ng
        if ! command -v aws >/dev/null 2>&1; then
          dnf install -y awscli
        fi

        mkdir -p "$RUN_ROOT/assets" "$DATA_DIR" "$WORKSPACE_DIR"

        aws s3 cp "$SUITE_ASSET_URI" "$RUN_ROOT/assets/aws_ultra_suite.zip" --no-progress
        rm -rf "$SUITE_DIR"
        mkdir -p "$SUITE_DIR"
        unzip -qo "$RUN_ROOT/assets/aws_ultra_suite.zip" -d "$SUITE_DIR"
        chmod +x "$SUITE_DIR/aws_bootstrap.sh"

        if [[ -n "$DATA_ASSET_URI" ]]; then
          aws s3 cp "$DATA_ASSET_URI" "$RUN_ROOT/assets/data_bundle.zip" --no-progress
          rm -rf "$DATA_DIR"
          mkdir -p "$DATA_DIR"
          unzip -qo "$RUN_ROOT/assets/data_bundle.zip" -d "$DATA_DIR"
        fi

        cat > "$SUITE_DIR/config.runtime.json" <<'JSON'
        {runtime_json}
        JSON

        RUN_EXIT_CODE=0
        bash "$SUITE_DIR/aws_bootstrap.sh" "$SUITE_DIR/config.runtime.json" {deploy_config.profile} || RUN_EXIT_CODE=$?

        mkdir -p "$RUN_ROOT/final"
        printf '%s\\n' "$RUN_EXIT_CODE" > "$RUN_ROOT/final/exit_code.txt"
        date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/final/finished_at_utc.txt"

        aws s3 sync "$WORKSPACE_DIR" "$RESULTS_S3_URI/workspace/" --no-progress || true
        aws s3 sync "$RUN_ROOT/final" "$RESULTS_S3_URI/final/" --no-progress || true
        aws s3 cp "$SUITE_DIR/config.runtime.json" "$RESULTS_S3_URI/bootstrap/config.runtime.json" --no-progress || true
        aws s3 cp /var/log/ultra-runner-user-data.log "$RESULTS_S3_URI/bootstrap/ultra-runner-user-data.log" --no-progress || true

        if [[ "$RUN_EXIT_CODE" -eq 0 && "$TERMINATE_ON_SUCCESS" == "true" ]]; then
          shutdown -h now
        fi

        if [[ "$RUN_EXIT_CODE" -ne 0 && "$TERMINATE_ON_FAILURE" == "true" ]]; then
          shutdown -h now
        fi

        exit "$RUN_EXIT_CODE"
        """
    ).strip()
