from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODEL_MODULES = {
    "knn": "model_knn",
    "svm": "model_svm",
    "random_forest": "model_random_forest",
    "bagging": "model_bagging",
}


def configure_env_for_model(model_key: str) -> None:
    if model_key == "svm":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    else:
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AWS-optimized ultra model.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_MODULES))
    parser.add_argument("--config", default=None, help="Path to config JSON.")
    parser.add_argument("--profile", default=None, help="Override profile from config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_env_for_model(args.model)

    module_name = MODEL_MODULES[args.model]
    suite_root = Path(__file__).resolve().parent
    if str(suite_root) not in sys.path:
        sys.path.insert(0, str(suite_root))

    from common import load_config, maybe_sync_data_from_s3, stage_log  # imported after env tuning

    model_module = __import__(module_name)
    config = load_config(args.config, profile_override=args.profile)

    maybe_sync_data_from_s3(config)
    stage_log(f"Dispatching model={args.model} with profile={config.profile}")
    summary_path = model_module.run(config)
    stage_log(f"Model {args.model} finished. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
