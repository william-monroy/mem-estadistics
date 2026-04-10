#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import aws_cdk as cdk

from ultra_runner_cdk.config import load_deploy_config
from ultra_runner_cdk.stack import UltraRunnerStack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the AWS Ultra Suite runner stack.")
    parser.add_argument(
        "--config",
        default=os.environ.get("ULTRA_RUNNER_DEPLOY_CONFIG"),
        help="Path to the deploy config JSON. Defaults to ULTRA_RUNNER_DEPLOY_CONFIG or deploy_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deploy_config = load_deploy_config(args.config)

    app = cdk.App()
    account = os.environ.get("CDK_DEFAULT_ACCOUNT")
    region = os.environ.get("CDK_DEFAULT_REGION")
    env = cdk.Environment(account=account, region=region) if account and region else None

    stack = UltraRunnerStack(
        app,
        deploy_config.stack_name,
        deploy_config=deploy_config,
        env=env,
        description="EC2 runner stack for challenge/aws_ultra_suite with S3 persistence and easy cleanup.",
    )

    cdk.Tags.of(stack).add("ManagedBy", "aws-cdk")
    cdk.Tags.of(stack).add("Application", "aws-ultra-suite")
    for key, value in deploy_config.tags.items():
        cdk.Tags.of(stack).add(key, value)

    app.synth()


if __name__ == "__main__":
    main()
