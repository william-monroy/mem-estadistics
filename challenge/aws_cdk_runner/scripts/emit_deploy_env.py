#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit shell exports derived from the deploy config.")
    parser.add_argument("--config", default=None, help="Path to the deploy config JSON.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from ultra_runner_cdk.config import load_deploy_config

    config = load_deploy_config(args.config)
    exports = {
        "ULTRA_CDK_STACK_NAME": config.stack_name,
        "ULTRA_CDK_AWS_PROFILE": config.aws_profile or "",
        "ULTRA_CDK_AWS_REGION": config.aws_region or "",
        "ULTRA_CDK_CONFIG_PATH": str(config.config_path),
    }
    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
