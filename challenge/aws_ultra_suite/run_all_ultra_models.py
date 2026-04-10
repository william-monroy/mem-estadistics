from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL_ORDER_DEFAULT = ["knn", "svm", "bagging", "random_forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AWS ultra suite sequentially.")
    parser.add_argument("--config", default=None, help="Path to config JSON.")
    parser.add_argument("--profile", default=None, help="Override profile from config.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=MODEL_ORDER_DEFAULT,
        help="Subset of models to run. Default: all.",
    )
    return parser.parse_args()


def load_config_payload(config_path: str | None) -> dict:
    if not config_path:
        return {}
    return json.loads(Path(config_path).read_text())


def stream_subprocess(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    suite_root = Path(__file__).resolve().parent
    if str(suite_root) not in sys.path:
        sys.path.insert(0, str(suite_root))

    from common import cpu_runtime_snapshot, load_config, maybe_sync_path_to_s3, stage_log, write_global_summary_csv

    config = load_config(args.config, profile_override=args.profile)
    payload = load_config_payload(args.config)

    model_list = args.models or payload.get("models") or MODEL_ORDER_DEFAULT
    continue_on_error = bool(payload.get("continue_on_error", config.continue_on_error))

    stage_log(f"AWS ultra suite starting. Runtime snapshot -> {cpu_runtime_snapshot(config)}")
    stage_log(f"Model order -> {model_list}")

    run_records: list[dict] = []
    for model_key in model_list:
        start_time = time.time()
        log_path = config.logs_root / f"{model_key}.log"
        command = [
            sys.executable,
            str(suite_root / "run_single_model.py"),
            "--model",
            model_key,
        ]
        if args.config:
            command.extend(["--config", args.config])
        if args.profile:
            command.extend(["--profile", args.profile])

        stage_log(f"Launching model={model_key}. Log -> {log_path}")
        exit_code = stream_subprocess(command, log_path)
        elapsed_minutes = round((time.time() - start_time) / 60.0, 3)

        summary_path = None
        validation_accuracy = None
        submission_path = None
        model_name = model_key

        candidate_summaries = sorted(config.output_root.glob(f"*/summary.json"))
        for path in candidate_summaries:
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if payload.get("model_key") == model_key:
                summary_path = str(path)
                validation_accuracy = payload.get("validation_accuracy")
                submission_path = payload.get("submission_path")
                model_name = payload.get("model_name", model_name)
                break

        run_records.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "exit_code": exit_code,
                "elapsed_minutes": elapsed_minutes,
                "summary_path": summary_path,
                "validation_accuracy": validation_accuracy,
                "submission_path": submission_path,
                "log_path": str(log_path),
            }
        )

        write_global_summary_csv(config.workspace_root, run_records)

        if exit_code != 0:
            stage_log(f"Model {model_key} failed with exit_code={exit_code}")
            if not continue_on_error:
                break
        else:
            stage_log(f"Model {model_key} completed in {elapsed_minutes} minutes")

    json_summary_path = config.workspace_root / "aws_ultra_run_summary.json"
    json_summary_path.write_text(json.dumps(run_records, indent=2))
    stage_log(f"Run summary written to {json_summary_path}")

    if config.s3_output_uri:
        maybe_sync_path_to_s3(config, config.workspace_root, config.s3_output_uri.rstrip("/") + "/workspace/")

    final_exit_code = 0 if all(record["exit_code"] == 0 for record in run_records) else 1
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
