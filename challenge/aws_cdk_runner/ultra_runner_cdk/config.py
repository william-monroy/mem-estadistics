from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


ALLOWED_MODELS = ["knn", "svm", "bagging", "random_forest"]


@dataclass
class DeployConfig:
    config_path: Path
    stack_name: str = "UltraRunnerStack"
    aws_profile: str | None = None
    aws_region: str | None = None
    instance_type: str = "c7a.8xlarge"
    volume_size_gb: int = 200
    profile: str = "aws_fast"
    models: list[str] = field(default_factory=lambda: ALLOWED_MODELS.copy())
    continue_on_error: bool = False
    sync_outputs_every_stage: bool = True
    create_resume_bundles: bool = True
    include_data_in_bundles: bool = False
    include_data_asset: bool = True
    local_suite_dir: Path = Path("../aws_ultra_suite")
    local_data_dir: Path = Path("../data")
    s3_data_uri: str | None = None
    reserve_cpus: int = 2
    max_cpu_fraction: float = 0.95
    svm_parallel_jobs: int | None = None
    svm_cache_mb: int | None = None
    knn_n_jobs: int | None = None
    ensemble_n_jobs: int | None = None
    terminate_on_success: bool = True
    terminate_on_failure: bool = False
    results_prefix_base: str = "runs"
    results_bucket_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.config_path.parent

    @property
    def results_prefix(self) -> str:
        base = self.results_prefix_base.strip("/")
        if not base:
            base = "runs"
        return f"{base}/{self.stack_name}"


def _runner_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    root = _runner_root()
    preferred = root / "deploy_config.json"
    if preferred.exists():
        return preferred
    return root / "deploy_config.example.json"


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _clean_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_deploy_config(config_path: str | os.PathLike[str] | None = None) -> DeployConfig:
    if config_path:
        config_file = Path(config_path).expanduser()
        if not config_file.is_absolute():
            config_file = (Path.cwd() / config_file).resolve()
        else:
            config_file = config_file.resolve()
    else:
        config_file = default_config_path().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Deploy config not found: {config_file}")

    payload = json.loads(config_file.read_text())
    allowed_keys = {
        "stack_name",
        "aws_profile",
        "aws_region",
        "instance_type",
        "volume_size_gb",
        "profile",
        "models",
        "continue_on_error",
        "sync_outputs_every_stage",
        "create_resume_bundles",
        "include_data_in_bundles",
        "include_data_asset",
        "local_suite_dir",
        "local_data_dir",
        "s3_data_uri",
        "reserve_cpus",
        "max_cpu_fraction",
        "svm_parallel_jobs",
        "svm_cache_mb",
        "knn_n_jobs",
        "ensemble_n_jobs",
        "terminate_on_success",
        "terminate_on_failure",
        "results_prefix_base",
        "results_bucket_name",
        "tags",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown keys in deploy config: {unknown_keys}")

    models = payload.get("models", ALLOWED_MODELS)
    if not isinstance(models, list) or not models:
        raise ValueError("'models' must be a non-empty list.")
    models = list(dict.fromkeys(str(model) for model in models))
    invalid_models = sorted(set(models) - set(ALLOWED_MODELS))
    if invalid_models:
        raise ValueError(f"Unsupported models in deploy config: {invalid_models}")

    raw_tags = payload.get("tags") or {}
    if not isinstance(raw_tags, dict):
        raise ValueError("'tags' must be an object/dict when provided.")

    base_dir = config_file.parent
    suite_dir = _resolve_path(base_dir, payload.get("local_suite_dir", "../aws_ultra_suite"))
    data_dir = _resolve_path(base_dir, payload.get("local_data_dir", "../data"))

    config = DeployConfig(
        config_path=config_file,
        stack_name=str(payload.get("stack_name", "UltraRunnerStack")),
        aws_profile=_clean_optional_string(payload.get("aws_profile")),
        aws_region=_clean_optional_string(payload.get("aws_region")),
        instance_type=str(payload.get("instance_type", "c7a.8xlarge")),
        volume_size_gb=int(payload.get("volume_size_gb", 200)),
        profile=str(payload.get("profile", "aws_fast")),
        models=models,
        continue_on_error=bool(payload.get("continue_on_error", False)),
        sync_outputs_every_stage=bool(payload.get("sync_outputs_every_stage", True)),
        create_resume_bundles=bool(payload.get("create_resume_bundles", True)),
        include_data_in_bundles=bool(payload.get("include_data_in_bundles", False)),
        include_data_asset=bool(payload.get("include_data_asset", True)),
        local_suite_dir=suite_dir,
        local_data_dir=data_dir,
        s3_data_uri=_clean_optional_string(payload.get("s3_data_uri")),
        reserve_cpus=int(payload.get("reserve_cpus", 2)),
        max_cpu_fraction=float(payload.get("max_cpu_fraction", 0.95)),
        svm_parallel_jobs=payload.get("svm_parallel_jobs"),
        svm_cache_mb=payload.get("svm_cache_mb"),
        knn_n_jobs=payload.get("knn_n_jobs"),
        ensemble_n_jobs=payload.get("ensemble_n_jobs"),
        terminate_on_success=bool(payload.get("terminate_on_success", True)),
        terminate_on_failure=bool(payload.get("terminate_on_failure", False)),
        results_prefix_base=str(payload.get("results_prefix_base", "runs")),
        results_bucket_name=_clean_optional_string(payload.get("results_bucket_name")),
        tags={str(k): str(v) for k, v in raw_tags.items()},
    )

    if not config.local_suite_dir.exists():
        raise FileNotFoundError(f"local_suite_dir does not exist: {config.local_suite_dir}")
    if not config.local_suite_dir.is_dir():
        raise ValueError(f"local_suite_dir must be a directory: {config.local_suite_dir}")

    if config.include_data_asset:
        if not config.local_data_dir.exists():
            raise FileNotFoundError(f"local_data_dir does not exist: {config.local_data_dir}")
        if not config.local_data_dir.is_dir():
            raise ValueError(f"local_data_dir must be a directory: {config.local_data_dir}")
    elif not config.s3_data_uri:
        raise ValueError("Set include_data_asset=true or provide s3_data_uri.")

    if config.volume_size_gb < 40:
        raise ValueError("volume_size_gb must be at least 40 GB.")
    if not (0.1 <= config.max_cpu_fraction <= 1.0):
        raise ValueError("max_cpu_fraction must be between 0.1 and 1.0.")

    return config
