from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SUITE_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _SUITE_DIR / ".cache" / "matplotlib"
_XDG_CACHE_DIR = _SUITE_DIR / ".cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["savefig.bbox"] = "tight"


DEFAULT_RANDOM_STATE = 301655
DEFAULT_VALID_SIZE = 0.20


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class AwsUltraConfig:
    workspace_root: Path
    data_dir: Path
    output_root: Path
    submissions_root: Path
    exports_root: Path
    logs_root: Path
    temp_root: Path
    profile: str = "aws_fast"
    models: list[str] = field(default_factory=lambda: ["knn", "svm", "bagging", "random_forest"])
    continue_on_error: bool = False
    valid_size: float = DEFAULT_VALID_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    reserve_cpus: int = 2
    max_cpu_fraction: float = 0.95
    svm_parallel_jobs: int | None = None
    svm_cache_mb: int | None = None
    knn_n_jobs: int | None = None
    ensemble_n_jobs: int | None = None
    s3_data_uri: str | None = None
    s3_output_uri: str | None = None
    sync_outputs_every_stage: bool = False
    create_resume_bundles: bool = True
    include_data_in_bundles: bool = False

    @property
    def cpu_count(self) -> int:
        return os.cpu_count() or 2

    @property
    def ram_gb(self) -> float | None:
        try:
            import psutil

            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            return None

    def auto_cpu_budget(self) -> int:
        scaled = math.floor(self.cpu_count * self.max_cpu_fraction)
        return max(1, scaled - self.reserve_cpus)

    def resolved_knn_n_jobs(self) -> int:
        return self.knn_n_jobs or self.auto_cpu_budget()

    def resolved_ensemble_n_jobs(self) -> int:
        return self.ensemble_n_jobs or self.auto_cpu_budget()

    def resolved_svm_parallel_jobs(self) -> int:
        return self.svm_parallel_jobs or max(1, min(8, self.auto_cpu_budget()))

    def resolved_svm_cache_mb(self) -> int:
        if self.svm_cache_mb is not None:
            return self.svm_cache_mb
        ram_gb = self.ram_gb or 32
        if ram_gb >= 96:
            return 6144
        if ram_gb >= 64:
            return 4096
        if ram_gb >= 32:
            return 2048
        return 1024


@dataclass
class ModelContext:
    config: AwsUltraConfig
    model_key: str
    model_name: str
    slug: str
    output_dir: Path
    checkpoint_dir: Path
    export_dir: Path
    log_dir: Path


def detect_suite_root() -> Path:
    return Path(__file__).resolve().parent


def detect_repo_root() -> Path:
    return detect_suite_root().parents[1]


def load_config(config_path: str | Path | None = None, *, profile_override: str | None = None) -> AwsUltraConfig:
    suite_root = detect_suite_root()
    repo_root = detect_repo_root()
    config_base_dir = suite_root

    default_workspace = suite_root / "workspace"
    default_payload: dict[str, Any] = {
        "workspace_root": str(default_workspace),
        "data_dir": str(repo_root / "challenge" / "data"),
        "output_root": str(default_workspace / "output"),
        "submissions_root": str(default_workspace / "submissions"),
        "exports_root": str(default_workspace / "exports"),
        "logs_root": str(default_workspace / "logs"),
        "temp_root": str(default_workspace / "tmp"),
        "profile": "aws_fast",
        "models": ["knn", "svm", "bagging", "random_forest"],
        "continue_on_error": False,
        "valid_size": DEFAULT_VALID_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "reserve_cpus": 2,
        "max_cpu_fraction": 0.95,
        "svm_parallel_jobs": None,
        "svm_cache_mb": None,
        "knn_n_jobs": None,
        "ensemble_n_jobs": None,
        "s3_data_uri": None,
        "s3_output_uri": None,
        "sync_outputs_every_stage": False,
        "create_resume_bundles": True,
        "include_data_in_bundles": False,
    }

    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        config_base_dir = config_path.parent
        config_payload = json.loads(config_path.read_text())
        default_payload.update(config_payload)

    if profile_override is not None:
        default_payload["profile"] = profile_override

    for key in ["workspace_root", "data_dir", "output_root", "submissions_root", "exports_root", "logs_root", "temp_root"]:
        candidate = Path(default_payload[key]).expanduser()
        if not candidate.is_absolute():
            candidate = (config_base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        default_payload[key] = candidate

    config = AwsUltraConfig(**default_payload)
    ensure_workspace(config)
    return config


def ensure_workspace(config: AwsUltraConfig) -> None:
    for path in [
        config.workspace_root,
        config.data_dir,
        config.output_root,
        config.submissions_root,
        config.exports_root,
        config.logs_root,
        config.temp_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_model_context(config: AwsUltraConfig, model_key: str, slug: str, model_name: str) -> ModelContext:
    output_dir = config.output_root / slug
    checkpoint_dir = output_dir / "checkpoints"
    export_dir = config.exports_root / slug
    log_dir = config.logs_root / slug

    for path in [output_dir, checkpoint_dir, export_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return ModelContext(
        config=config,
        model_key=model_key,
        model_name=model_name,
        slug=slug,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
        log_dir=log_dir,
    )


def ensure_required_csvs(config: AwsUltraConfig) -> None:
    missing = []
    for name in ["training.csv", "test.csv", "sample.csv"]:
        path = config.data_dir / name
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing required CSV files. "
            "Place training.csv, test.csv and sample.csv in the configured data_dir "
            f"or configure s3_data_uri. Missing: {missing}"
        )


def run_aws_cli(args: list[str]) -> None:
    executable = shutil.which("aws")
    if executable is None:
        raise RuntimeError("AWS CLI is not available in PATH. Install it or disable S3 sync.")

    command = [executable, *args]
    subprocess.run(command, check=True)


def maybe_sync_data_from_s3(config: AwsUltraConfig) -> None:
    if not config.s3_data_uri:
        return
    run_aws_cli(["s3", "sync", config.s3_data_uri, str(config.data_dir), "--no-progress"])


def maybe_sync_path_to_s3(config: AwsUltraConfig, local_path: Path, s3_uri: str) -> None:
    if not s3_uri:
        return
    if local_path.is_dir():
        run_aws_cli(["s3", "sync", str(local_path), s3_uri, "--no-progress"])
    else:
        run_aws_cli(["s3", "cp", str(local_path), s3_uri, "--no-progress"])


def maybe_sync_model_outputs_to_s3(config: AwsUltraConfig, context: ModelContext) -> None:
    if not config.s3_output_uri:
        return
    base_uri = config.s3_output_uri.rstrip("/")
    maybe_sync_path_to_s3(config, context.output_dir, f"{base_uri}/output/{context.slug}/")
    maybe_sync_path_to_s3(config, context.export_dir, f"{base_uri}/exports/{context.slug}/")


def save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def read_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    sort_cols: list[str] | None = None,
    ascending: bool | list[bool] = True,
) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    current = read_dataframe(path)
    combined = new_df if current.empty else pd.concat([current, new_df], ignore_index=True)
    if sort_cols:
        combined = combined.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    save_dataframe_atomic(combined, path)
    return combined


def update_manifest(context: ModelContext, extra_payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = context.checkpoint_dir / "manifest.json"
    manifest = read_json(manifest_path, default={})
    manifest.update(extra_payload)
    write_json_atomic(manifest_path, manifest)
    return manifest


def create_resume_bundle(context: ModelContext, *, include_data: bool) -> Path:
    bundle_path = context.export_dir / f"{context.slug}_resume.zip"
    if bundle_path.exists():
        bundle_path.unlink()

    paths_to_pack: list[Path] = []
    if include_data:
        paths_to_pack.append(context.config.data_dir)
    paths_to_pack.extend([context.output_dir, context.config.submissions_root])

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for root_path in paths_to_pack:
            if not root_path.exists():
                continue
            for nested in root_path.rglob("*"):
                if nested.is_dir():
                    continue
                relative = nested.relative_to(context.config.workspace_root)
                zip_file.write(nested, arcname=str(relative))

    return bundle_path


def checkpoint_housekeeping(
    context: ModelContext,
    stage_name: str,
    *,
    refresh_bundle: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "last_checkpoint_stage": stage_name,
        "timestamp_epoch": time.time(),
    }

    if refresh_bundle and context.config.create_resume_bundles:
        bundle_path = create_resume_bundle(context, include_data=context.config.include_data_in_bundles)
        payload["resume_bundle_path"] = str(bundle_path)
        payload["resume_bundle_size_mb"] = round(bundle_path.stat().st_size / (1024 ** 2), 3)

    manifest = update_manifest(context, payload)

    if context.config.sync_outputs_every_stage:
        maybe_sync_model_outputs_to_s3(context.config, context)

    return manifest


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }


def load_base_data(config: AwsUltraConfig) -> dict[str, Any]:
    ensure_required_csvs(config)
    train_df = pd.read_csv(config.data_dir / "training.csv")
    test_df = pd.read_csv(config.data_dir / "test.csv")
    sample_df = pd.read_csv(config.data_dir / "sample.csv")

    feature_names = [column for column in train_df.columns if column not in {"id", "class"}]
    X_full = train_df[feature_names].astype(np.float32).to_numpy()
    y_full = train_df["class"].astype(np.int8).to_numpy()
    X_test_full = test_df[feature_names].astype(np.float32).to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full,
        y_full,
        test_size=config.valid_size,
        stratify=y_full,
        random_state=config.random_state,
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "sample_df": sample_df,
        "feature_names": feature_names,
        "X_full": X_full,
        "y_full": y_full,
        "X_test_full": X_test_full,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
    }


def render_validation_confusion_matrix(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> Path:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Undamaged (0)", "Damaged (1)"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title(title)
    return save_current_figure(output_dir / "validation_confusion_matrix.png")


def stage_log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_model_summary(context: ModelContext, payload: dict[str, Any]) -> Path:
    summary_path = context.output_dir / "summary.json"
    write_json_atomic(summary_path, payload)
    return summary_path


def write_global_summary_csv(workspace_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = workspace_root / "aws_ultra_model_summary.csv"
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe_atomic(df, path)
    return path


def cpu_runtime_snapshot(config: AwsUltraConfig) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "cpu_count": config.cpu_count,
        "ram_gb": config.ram_gb,
        "profile": config.profile,
        "workspace_root": str(config.workspace_root),
        "data_dir": str(config.data_dir),
    }
