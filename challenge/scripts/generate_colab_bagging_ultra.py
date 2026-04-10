from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "colab_bagging_ultra"
NOTEBOOK_PATH = TARGET_DIR / "Challenge_07_Bagging_Colab_Ultra.ipynb"


INTRO = dedent(
    """
    # Challenge 07 Colab Ultra: Bagging over decision trees

    ## Objective

    This notebook is a Google Colab oriented replacement for a deep Bagging search based on decision trees.
    It is designed to be:

    1. Faster on CPU runtimes by using staged screening instead of a full Cartesian grid.
    2. Safe to start from a blank Colab session where only this notebook has been uploaded.
    3. Re-runnable with checkpoints so you do not lose the entire search after a disconnect.

    ## Search strategy

    The search is intentionally divided into four stages:

    1. Stage 1 screens a broad random sample with out-of-bag score on a reduced training subset.
    2. Stage 2 promotes the top candidates to 3-fold cross-validation on the full training split.
    3. Stage 3 grows only the finalists through a larger ensemble schedule using `warm_start=True`.
    4. Stage 4 performs a narrow local refinement around the best Stage 3 configuration.

    ## Colab-specific design decisions

    - CPU runtime, not GPU. `scikit-learn` bagging ensembles over trees do not meaningfully benefit from standard Colab GPU runtimes.
    - The base workflow assumes a blank runtime and manual uploads done from inside Colab.
    - A runtime workspace is created automatically under `/content`.
    - Google Drive persistence is optional, not required.
    - Resume logic is based on checkpoint CSV and JSON files, plus an exportable resume bundle.
    """
).strip()


IMPORTS = dedent(
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import gc
    import json
    import math
    import platform
    import shutil
    import time
    import warnings
    import zipfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sklearn
    from IPython.display import display
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
    from sklearn.tree import DecisionTreeClassifier

    try:
        import psutil
    except ImportError:
        psutil = None

    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(iterable=None, **kwargs):
            return iterable if iterable is not None else []

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.bbox"] = "tight"

    RANDOM_STATE = 301655
    VALID_SIZE = 0.20
    NOTEBOOK_SLUG = "challenge_07_bagging_colab_ultra"
    """
).strip()


COLAB_SETUP = dedent(
    """
    try:
        import google.colab  # type: ignore
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        from google.colab import files  # type: ignore
    else:
        files = None

    if IN_COLAB:
        WORKSPACE_ROOT = Path("/content/challenge_bagging_ultra_workspace")
    else:
        cwd = Path.cwd().resolve()
        if (cwd / "challenge" / "data" / "training.csv").exists():
            WORKSPACE_ROOT = cwd / "challenge"
        elif (cwd / "data" / "training.csv").exists():
            WORKSPACE_ROOT = cwd
        else:
            WORKSPACE_ROOT = cwd / "challenge_bagging_ultra_workspace"

    DATA_DIR = WORKSPACE_ROOT / "data"
    TRAIN_PATH = DATA_DIR / "training.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    SAMPLE_PATH = DATA_DIR / "sample.csv"

    OUTPUT_ROOT = WORKSPACE_ROOT / "output"
    PERSIST_ROOT = OUTPUT_ROOT / NOTEBOOK_SLUG
    CHECKPOINT_DIR = PERSIST_ROOT / "checkpoints"
    SUBMISSION_DIR = WORKSPACE_ROOT / "submissions"
    EXPORT_DIR = WORKSPACE_ROOT / "exports"

    for path in [WORKSPACE_ROOT, DATA_DIR, OUTPUT_ROOT, PERSIST_ROOT, CHECKPOINT_DIR, SUBMISSION_DIR, EXPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    ENABLE_GOOGLE_DRIVE_PERSISTENCE = False
    DRIVE_PERSIST_ROOT = Path("/content/drive/MyDrive/challenge_bagging_ultra_workspace")

    print("IN_COLAB:", IN_COLAB)
    print("WORKSPACE_ROOT:", WORKSPACE_ROOT)
    print("PERSIST_ROOT:", PERSIST_ROOT)
    """
).strip()


WORKSPACE_STATUS = dedent(
    """
    expected_files = {
        "training.csv": TRAIN_PATH,
        "test.csv": TEST_PATH,
        "sample.csv": SAMPLE_PATH,
    }

    print("Workspace directories:")
    for path in [WORKSPACE_ROOT, DATA_DIR, OUTPUT_ROOT, PERSIST_ROOT, CHECKPOINT_DIR, SUBMISSION_DIR, EXPORT_DIR]:
        print("-", path)

    print("\\nData file status:")
    for filename, path in expected_files.items():
        print(f"- {filename}: {'OK' if path.exists() else 'MISSING'} -> {path}")

    print(
        "\\nUse one of the next optional cells to populate data/:\\n"
        "1. Upload training.csv, test.csv and sample.csv directly.\\n"
        "2. Upload a previously exported resume ZIP that already contains data/ and output/."
    )
    """
).strip()


UPLOAD_DATA = dedent(
    """
    UPLOAD_DATA_FILES = False

    if UPLOAD_DATA_FILES:
        if not IN_COLAB:
            raise RuntimeError("This upload helper is only intended for Google Colab.")

        uploaded = files.upload()
        for original_name, file_bytes in uploaded.items():
            filename = Path(original_name).name
            target_path = DATA_DIR / filename
            target_path.write_bytes(file_bytes)
            print("Saved:", target_path)
    else:
        print("Set UPLOAD_DATA_FILES = True if you want to upload the CSV files from your browser.")
    """
).strip()


UPLOAD_RESUME_BUNDLE = dedent(
    """
    RESTORE_RESUME_BUNDLE = False

    if RESTORE_RESUME_BUNDLE:
        if not IN_COLAB:
            raise RuntimeError("This resume-bundle helper is only intended for Google Colab.")

        uploaded = files.upload()
        zip_names = [Path(name).name for name in uploaded if str(name).lower().endswith(".zip")]
        if len(zip_names) != 1:
            raise ValueError("Upload exactly one ZIP file when restoring a resume bundle.")

        bundle_name = zip_names[0]
        bundle_path = EXPORT_DIR / bundle_name
        bundle_path.write_bytes(uploaded[bundle_name])

        with zipfile.ZipFile(bundle_path, "r") as zip_file:
            zip_file.extractall(WORKSPACE_ROOT)

        print("Resume bundle restored into:", WORKSPACE_ROOT)
    else:
        print("Set RESTORE_RESUME_BUNDLE = True if you want to restore a previous ZIP bundle.")
    """
).strip()


OPTIONAL_DRIVE = dedent(
    """
    if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
        if not IN_COLAB:
            raise RuntimeError("Google Drive persistence is only available inside Google Colab.")

        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        DRIVE_PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
        restore_workspace_from_drive()
        print("Drive persistence enabled at:", DRIVE_PERSIST_ROOT)
        print("Any previously saved workspace files have been copied back into the runtime workspace.")
    else:
        print("Drive persistence disabled. The notebook will use only the Colab runtime workspace.")
    """
).strip()


VALIDATE_DATA = dedent(
    """
    missing_files = [str(path) for path in [TRAIN_PATH, TEST_PATH, SAMPLE_PATH] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required data files in the Colab workspace.\\n"
            "Upload training.csv, test.csv and sample.csv into data/ or restore a resume ZIP first.\\n"
            f"Missing: {missing_files}"
        )

    print("All required CSV files are present.")
    """
).strip()


RUNTIME_CONFIG = dedent(
    """
    CPU_COUNT = os.cpu_count() or 2
    RAM_GB = None if psutil is None else round(psutil.virtual_memory().total / (1024 ** 3), 2)
    BAGGING_N_JOBS = max(1, CPU_COUNT - 1)

    SEARCH_PRESETS = {
        "balanced": {
            "stage1_n_iter": 72,
            "stage1_train_fraction": 0.70,
            "stage1_n_estimators": 128,
            "stage2_top_k": 16,
            "stage2_cv": 3,
            "stage2_n_estimators": 256,
            "stage3_top_k": 4,
            "stage3_cv": 5,
            "stage3_tree_schedule": [256, 384, 512, 768],
            "stage4_cv": 5,
        },
        "aggressive": {
            "stage1_n_iter": 96,
            "stage1_train_fraction": 0.80,
            "stage1_n_estimators": 160,
            "stage2_top_k": 18,
            "stage2_cv": 3,
            "stage2_n_estimators": 320,
            "stage3_top_k": 5,
            "stage3_cv": 5,
            "stage3_tree_schedule": [320, 512, 768, 1024],
            "stage4_cv": 5,
        },
    }

    SEARCH_PROFILE = "aggressive"
    PROFILE = SEARCH_PRESETS[SEARCH_PROFILE]

    RUN_STAGE_1 = True
    RUN_STAGE_2 = True
    RUN_STAGE_3 = True
    RUN_STAGE_4 = True
    TRAIN_FINAL_MODEL = True

    print(
        {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "cpu_count": CPU_COUNT,
            "ram_gb": RAM_GB,
            "bagging_n_jobs": BAGGING_N_JOBS,
            "search_profile": SEARCH_PROFILE,
        }
    )
    """
).strip()


LOAD_DATA = dedent(
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_df = pd.read_csv(SAMPLE_PATH)

    feature_names = [column for column in train_df.columns if column not in {"id", "class"}]
    X_full = train_df[feature_names].astype(np.float32).to_numpy()
    y_full = train_df["class"].astype(np.int8).to_numpy()
    X_test_full = test_df[feature_names].astype(np.float32).to_numpy()

    print("Training shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Class balance:", train_df["class"].value_counts().sort_index().to_dict())
    print("Missing values in training:", int(train_df.isna().sum().sum()))
    print("Duplicated rows in training:", int(train_df.duplicated().sum()))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full,
        y_full,
        test_size=VALID_SIZE,
        stratify=y_full,
        random_state=RANDOM_STATE,
    )

    if PROFILE["stage1_train_fraction"] < 1.0:
        X_stage1, _, y_stage1, _ = train_test_split(
            X_train,
            y_train,
            train_size=PROFILE["stage1_train_fraction"],
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_stage1, y_stage1 = X_train, y_train

    print("Train split:", X_train.shape, y_train.shape)
    print("Validation split:", X_valid.shape, y_valid.shape)
    print("Stage 1 subset:", X_stage1.shape, y_stage1.shape)
    """
).strip()


HELPERS = dedent(
    """
    def save_current_figure(filename: str) -> Path:
        path = PERSIST_ROOT / filename
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        return path


    def write_json_atomic(path: Path, payload: dict | list) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)


    def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)


    def read_dataframe(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


    def copy_path(src: Path, dst: Path) -> None:
        if src.is_dir():
            for nested in src.rglob("*"):
                relative = nested.relative_to(src)
                target = dst / relative
                if nested.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(nested, target)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


    def normalize_value(value):
        if isinstance(value, np.generic):
            return value.item()
        return value


    def normalize_candidate(params: dict) -> dict:
        normalized = {key: normalize_value(value) for key, value in params.items()}
        if normalized.get("base__max_depth") is not None:
            normalized["base__max_depth"] = int(normalized["base__max_depth"])
        if normalized.get("base__min_samples_leaf") is not None:
            normalized["base__min_samples_leaf"] = int(normalized["base__min_samples_leaf"])
        if normalized.get("base__min_samples_split") is not None:
            normalized["base__min_samples_split"] = int(normalized["base__min_samples_split"])
        if normalized.get("model__max_features") is not None:
            normalized["model__max_features"] = float(normalized["model__max_features"])
        if normalized.get("model__max_samples") is not None:
            normalized["model__max_samples"] = float(normalized["model__max_samples"])
        normalized["model__bootstrap_features"] = bool(normalized["model__bootstrap_features"])
        return normalized


    def candidate_signature(params: dict) -> str:
        normalized = normalize_candidate(params)
        return json.dumps(normalized, sort_keys=True)


    def candidate_record(candidate_id: str, params: dict, source: str) -> dict:
        params = normalize_candidate(params)
        return {
            "candidate_id": candidate_id,
            "source": source,
            "params": params,
            "signature": candidate_signature(params),
        }


    def candidate_to_row(candidate: dict) -> dict:
        params = candidate["params"]
        return {
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "signature": candidate["signature"],
            "base__criterion": params.get("base__criterion"),
            "base__max_depth": params.get("base__max_depth"),
            "base__min_samples_leaf": params.get("base__min_samples_leaf"),
            "base__min_samples_split": params.get("base__min_samples_split"),
            "model__max_features": params.get("model__max_features"),
            "model__max_samples": params.get("model__max_samples"),
            "model__bootstrap_features": params.get("model__bootstrap_features"),
            "params_json": json.dumps(params, sort_keys=True),
        }


    def get_supported_criteria() -> list[str]:
        version_parts = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
        criteria = ["gini", "entropy"]
        if version_parts >= (1, 1):
            criteria.append("log_loss")
        return criteria


    TREE_CRITERIA = get_supported_criteria()
    print("Supported criteria:", TREE_CRITERIA)


    def build_base_tree(params: dict) -> DecisionTreeClassifier:
        params = normalize_candidate(params)
        return DecisionTreeClassifier(
            criterion=params["base__criterion"],
            max_depth=params["base__max_depth"],
            min_samples_leaf=params["base__min_samples_leaf"],
            min_samples_split=params["base__min_samples_split"],
            random_state=RANDOM_STATE,
        )


    def build_bagging(params: dict, n_estimators: int, random_state: int, *, oob_score: bool = False, warm_start: bool = False) -> BaggingClassifier:
        params = normalize_candidate(params)
        ensemble_kwargs = {
            "n_estimators": int(n_estimators),
            "bootstrap": True,
            "oob_score": oob_score,
            "warm_start": warm_start,
            "n_jobs": BAGGING_N_JOBS,
            "random_state": random_state,
            "max_samples": params["model__max_samples"],
            "max_features": params["model__max_features"],
            "bootstrap_features": params["model__bootstrap_features"],
        }
        tree = build_base_tree(params)
        try:
            return BaggingClassifier(estimator=tree, **ensemble_kwargs)
        except TypeError:
            return BaggingClassifier(base_estimator=tree, **ensemble_kwargs)


    def upsert_summary_row(path: Path, row: dict, key_cols: list[str]) -> pd.DataFrame:
        df = read_dataframe(path)
        row_df = pd.DataFrame([row])
        if df.empty:
            df = row_df
        else:
            mask = pd.Series(True, index=df.index)
            for key in key_cols:
                left = df[key].astype(str) if key in df else pd.Series("", index=df.index)
                mask &= left == str(row[key])
            df = pd.concat([df.loc[~mask], row_df], ignore_index=True)
        save_dataframe_atomic(df, path)
        return df


    def append_rows(path: Path, rows: list[dict], sort_cols: list[str] | None = None) -> pd.DataFrame:
        new_df = pd.DataFrame(rows)
        current = read_dataframe(path)
        if current.empty:
            combined = new_df
        else:
            combined = pd.concat([current, new_df], ignore_index=True)
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)
        save_dataframe_atomic(combined, path)
        return combined


    def update_manifest(extra_payload: dict) -> dict:
        manifest_path = CHECKPOINT_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update(extra_payload)
        write_json_atomic(manifest_path, manifest)
        return manifest


    def create_resume_bundle(bundle_name: str = "challenge_07_bagging_colab_ultra_resume.zip", include_data: bool = True) -> Path:
        bundle_path = EXPORT_DIR / bundle_name
        if bundle_path.exists():
            bundle_path.unlink()

        paths_to_pack = []
        if include_data:
            paths_to_pack.append(DATA_DIR)
        paths_to_pack.extend([OUTPUT_ROOT, SUBMISSION_DIR])

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for root_path in paths_to_pack:
                if not root_path.exists():
                    continue
                for nested in root_path.rglob("*"):
                    if nested.is_dir():
                        continue
                    relative_path = nested.relative_to(WORKSPACE_ROOT)
                    zip_file.write(nested, arcname=str(relative_path))

        return bundle_path


    def download_if_requested(path: Path, should_download: bool = False) -> None:
        if should_download and IN_COLAB:
            files.download(str(path))


    def sync_workspace_to_drive(relative_paths: list[Path] | None = None) -> None:
        if not ENABLE_GOOGLE_DRIVE_PERSISTENCE:
            return

        DRIVE_PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
        relative_paths = relative_paths or [Path("data"), Path("output"), Path("submissions"), Path("exports")]
        for relative in relative_paths:
            src = WORKSPACE_ROOT / relative
            dst = DRIVE_PERSIST_ROOT / relative
            if src.exists():
                copy_path(src, dst)


    def restore_workspace_from_drive(relative_paths: list[Path] | None = None) -> None:
        if not ENABLE_GOOGLE_DRIVE_PERSISTENCE:
            return

        relative_paths = relative_paths or [Path("data"), Path("output"), Path("submissions"), Path("exports")]
        for relative in relative_paths:
            src = DRIVE_PERSIST_ROOT / relative
            dst = WORKSPACE_ROOT / relative
            if src.exists():
                copy_path(src, dst)


    def checkpoint_housekeeping(stage_name: str, *, refresh_bundle: bool = False, include_data_in_bundle: bool = False) -> dict:
        payload = {"last_checkpoint_stage": stage_name}

        if refresh_bundle:
            bundle_path = create_resume_bundle(include_data=include_data_in_bundle)
            payload["resume_bundle_path"] = str(bundle_path)
            payload["resume_bundle_size_mb"] = round(bundle_path.stat().st_size / (1024 ** 2), 3)

        update_manifest(payload)

        if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
            sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG, Path("submissions"), Path("exports")])

        return payload


    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": cm.tolist(),
        }
    """
).strip()


STAGE1 = dedent(
    """
    STAGE1_SPACE = {
        "base__criterion": TREE_CRITERIA,
        "base__max_depth": [None, 8, 12, 16, 20, 28],
        "base__min_samples_leaf": [1, 2, 4, 6, 8],
        "base__min_samples_split": [2, 4, 8, 12, 16],
        "model__max_features": [0.25, 0.35, 0.50, 0.65, 0.80, 1.0],
        "model__max_samples": [0.50, 0.65, 0.80, 1.0],
        "model__bootstrap_features": [False, True],
    }

    stage1_candidates_path = CHECKPOINT_DIR / "stage1_candidates.json"
    stage1_results_path = CHECKPOINT_DIR / "stage1_oob_results.csv"


    def get_or_create_stage1_candidates() -> list[dict]:
        if stage1_candidates_path.exists():
            return json.loads(stage1_candidates_path.read_text())

        sampler = ParameterSampler(
            STAGE1_SPACE,
            n_iter=PROFILE["stage1_n_iter"],
            random_state=RANDOM_STATE,
        )

        seen = set()
        candidates = []
        for sampled_params in sampler:
            normalized = normalize_candidate(sampled_params)
            signature = candidate_signature(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            candidate_id = f"stage1_{len(candidates):03d}"
            candidates.append(candidate_record(candidate_id, normalized, "stage1_random_oob"))

        write_json_atomic(stage1_candidates_path, candidates)
        return candidates


    stage1_candidates = get_or_create_stage1_candidates()
    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    print("Stage 1 candidates:", len(stage1_candidates))
    print("Stage 1 already completed:", len(completed_stage1))

    if RUN_STAGE_1:
        iterable = [candidate for candidate in stage1_candidates if candidate["candidate_id"] not in completed_stage1]
        for candidate in tqdm(iterable, desc="Stage 1 OOB screening"):
            start_time = time.time()
            model = build_bagging(
                candidate["params"],
                n_estimators=PROFILE["stage1_n_estimators"],
                random_state=RANDOM_STATE,
                oob_score=True,
            )
            model.fit(X_stage1, y_stage1)
            elapsed = time.time() - start_time

            row = {
                **candidate_to_row(candidate),
                "stage": "stage1_oob",
                "n_estimators": PROFILE["stage1_n_estimators"],
                "oob_accuracy": float(model.oob_score_),
                "fit_seconds": round(elapsed, 3),
            }
            stage1_results = append_rows(stage1_results_path, [row], sort_cols=["oob_accuracy"])

            update_manifest(
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_oob_accuracy": float(stage1_results["oob_accuracy"].max()),
                }
            )
            if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
                sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG])
            del model
            gc.collect()

    stage1_results = read_dataframe(stage1_results_path)
    if not stage1_results.empty:
        stage1_results = stage1_results.sort_values(["oob_accuracy", "fit_seconds"], ascending=[False, True]).reset_index(drop=True)
        display(stage1_results.head(10))
        checkpoint_housekeeping("stage1_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STAGE2 = dedent(
    """
    stage2_fold_path = CHECKPOINT_DIR / "stage2_cv_fold_results.csv"
    stage2_summary_path = CHECKPOINT_DIR / "stage2_cv_summary.csv"

    if stage1_results.empty:
        raise RuntimeError("Stage 1 produced no results. Run Stage 1 before Stage 2.")

    stage2_shortlist = (
        stage1_results.sort_values(["oob_accuracy", "fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage2_top_k"])
        .copy()
    )

    stage2_candidates = []
    for row in stage2_shortlist.to_dict(orient="records"):
        stage2_candidates.append(
            candidate_record(
                candidate_id=row["candidate_id"],
                params=json.loads(row["params_json"]),
                source="stage2_shortlist_from_stage1",
            )
        )

    stage2_cv = StratifiedKFold(n_splits=PROFILE["stage2_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage2_splits = list(stage2_cv.split(X_train, y_train))


    def refresh_stage2_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage2_fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in stage2_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < PROFILE["stage2_cv"]:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": "stage2_cv",
                "n_estimators": PROFILE["stage2_n_estimators"],
                "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage2_summary_path)
        return summary_df


    stage2_summary = refresh_stage2_summary()
    completed_stage2 = set(stage2_summary["candidate_id"]) if not stage2_summary.empty else set()

    print("Stage 2 shortlisted candidates:", len(stage2_candidates))
    print("Stage 2 already completed:", len(completed_stage2))

    if RUN_STAGE_2:
        iterable = [candidate for candidate in stage2_candidates if candidate["candidate_id"] not in completed_stage2]
        for candidate in tqdm(iterable, desc="Stage 2 CV shortlist"):
            fold_df = read_dataframe(stage2_fold_path)
            done_folds = set(
                fold_df.loc[fold_df["candidate_id"] == candidate["candidate_id"], "fold_idx"].astype(int).tolist()
            ) if not fold_df.empty else set()

            for fold_idx, (train_idx, test_idx) in enumerate(stage2_splits):
                if fold_idx in done_folds:
                    continue

                start_time = time.time()
                model = build_bagging(
                    candidate["params"],
                    n_estimators=PROFILE["stage2_n_estimators"],
                    random_state=RANDOM_STATE + fold_idx,
                )
                model.fit(X_train[train_idx], y_train[train_idx])
                fold_pred = model.predict(X_train[test_idx])
                fold_accuracy = accuracy_score(y_train[test_idx], fold_pred)
                elapsed = time.time() - start_time

                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "fold_idx": fold_idx,
                    "n_estimators": PROFILE["stage2_n_estimators"],
                    "fold_accuracy": float(fold_accuracy),
                    "fit_seconds": round(elapsed, 3),
                }
                append_rows(stage2_fold_path, [row], sort_cols=["candidate_id", "fold_idx"])
                del model
                gc.collect()

            stage2_summary = refresh_stage2_summary()
            if not stage2_summary.empty:
                update_manifest(
                    {
                        "stage2_completed_candidates": int(stage2_summary["candidate_id"].nunique()),
                        "stage2_total_candidates": len(stage2_candidates),
                        "stage2_best_cv_accuracy": float(stage2_summary["cv_mean_accuracy"].max()),
                    }
                )
                if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
                    sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG])

    stage2_summary = read_dataframe(stage2_summary_path)
    if not stage2_summary.empty:
        display(stage2_summary.head(10))
        checkpoint_housekeeping("stage2_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STAGE3 = dedent(
    """
    stage3_fold_path = CHECKPOINT_DIR / "stage3_ensemble_growth_fold_results.csv"
    stage3_summary_path = CHECKPOINT_DIR / "stage3_ensemble_growth_summary.csv"

    if stage2_summary.empty:
        raise RuntimeError("Stage 2 produced no results. Run Stage 2 before Stage 3.")

    stage3_seed = (
        stage2_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage3_top_k"])
        .copy()
    )

    stage3_candidates = []
    for row in stage3_seed.to_dict(orient="records"):
        stage3_candidates.append(
            candidate_record(
                candidate_id=row["candidate_id"],
                params=json.loads(row["params_json"]),
                source="stage3_finalists_from_stage2",
            )
        )

    stage3_cv = StratifiedKFold(n_splits=PROFILE["stage3_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage3_splits = list(stage3_cv.split(X_train, y_train))
    tree_schedule = PROFILE["stage3_tree_schedule"]


    def refresh_stage3_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage3_fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in stage3_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if candidate_fold_df.empty:
                continue
            grouped = (
                candidate_fold_df.groupby("n_estimators", as_index=False)
                .agg(
                    cv_mean_accuracy=("fold_accuracy", "mean"),
                    cv_std_accuracy=("fold_accuracy", lambda x: float(np.std(x, ddof=0))),
                    total_fit_seconds=("fit_seconds", "sum"),
                    completed_folds=("fold_idx", "nunique"),
                )
            )
            grouped["candidate_id"] = candidate["candidate_id"]
            grouped["source"] = candidate["source"]
            grouped["signature"] = candidate["signature"]
            grouped["base__criterion"] = candidate["params"].get("base__criterion")
            grouped["base__max_depth"] = candidate["params"].get("base__max_depth")
            grouped["base__min_samples_leaf"] = candidate["params"].get("base__min_samples_leaf")
            grouped["base__min_samples_split"] = candidate["params"].get("base__min_samples_split")
            grouped["model__max_features"] = candidate["params"].get("model__max_features")
            grouped["model__max_samples"] = candidate["params"].get("model__max_samples")
            grouped["model__bootstrap_features"] = candidate["params"].get("model__bootstrap_features")
            grouped["params_json"] = json.dumps(candidate["params"], sort_keys=True)
            grouped["stage"] = "stage3_ensemble_growth"
            summary_rows.append(grouped)

        if not summary_rows:
            return pd.DataFrame()

        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
        save_dataframe_atomic(summary_df, stage3_summary_path)
        return summary_df


    stage3_summary = refresh_stage3_summary()
    print("Stage 3 finalists:", len(stage3_candidates))

    if RUN_STAGE_3:
        for candidate in tqdm(stage3_candidates, desc="Stage 3 ensemble growth"):
            fold_df = read_dataframe(stage3_fold_path)
            completed_folds = set()
            if not fold_df.empty:
                completed = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
                grouped = completed.groupby("fold_idx")["n_estimators"].nunique()
                completed_folds = set(grouped[grouped == len(tree_schedule)].index.astype(int).tolist())

            for fold_idx, (train_idx, test_idx) in enumerate(stage3_splits):
                if fold_idx in completed_folds:
                    continue

                model = build_bagging(
                    candidate["params"],
                    n_estimators=tree_schedule[0],
                    random_state=RANDOM_STATE + fold_idx,
                    warm_start=True,
                )

                fold_rows = []
                for n_estimators in tree_schedule:
                    start_time = time.time()
                    model.set_params(n_estimators=int(n_estimators))
                    model.fit(X_train[train_idx], y_train[train_idx])
                    fold_pred = model.predict(X_train[test_idx])
                    fold_accuracy = accuracy_score(y_train[test_idx], fold_pred)
                    elapsed = time.time() - start_time
                    fold_rows.append(
                        {
                            **candidate_to_row(candidate),
                            "stage": "stage3_ensemble_growth",
                            "fold_idx": fold_idx,
                            "n_estimators": int(n_estimators),
                            "fold_accuracy": float(fold_accuracy),
                            "fit_seconds": round(elapsed, 3),
                        }
                    )

                append_rows(stage3_fold_path, fold_rows, sort_cols=["candidate_id", "fold_idx", "n_estimators"])
                del model
                gc.collect()

            stage3_summary = refresh_stage3_summary()
            if not stage3_summary.empty:
                update_manifest(
                    {
                        "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                        "stage3_best_cv_accuracy": float(stage3_summary["cv_mean_accuracy"].max()),
                    }
                )
                if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
                    sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG])

    stage3_summary = read_dataframe(stage3_summary_path)
    if not stage3_summary.empty:
        display(stage3_summary.head(12))
        checkpoint_housekeeping("stage3_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STAGE4 = dedent(
    """
    stage4_candidates_path = CHECKPOINT_DIR / "stage4_local_candidates.json"
    stage4_fold_path = CHECKPOINT_DIR / "stage4_local_cv_fold_results.csv"
    stage4_summary_path = CHECKPOINT_DIR / "stage4_local_cv_summary.csv"

    if stage3_summary.empty:
        raise RuntimeError("Stage 3 produced no results. Run Stage 3 before Stage 4.")

    best_stage3_row = stage3_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).iloc[0]
    center_params = json.loads(best_stage3_row["params_json"])
    best_tree_count = int(best_stage3_row["n_estimators"])


    def dedupe_candidates(candidates: list[dict]) -> list[dict]:
        unique = []
        seen = set()
        for candidate in candidates:
            signature = candidate_signature(candidate["params"])
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(candidate)
        return unique


    def build_stage4_candidates() -> list[dict]:
        if stage4_candidates_path.exists():
            return json.loads(stage4_candidates_path.read_text())

        params = normalize_candidate(center_params)
        candidates = [candidate_record("stage4_000", params, "stage4_center_from_stage3")]

        max_depth_values = [params.get("base__max_depth")]
        if params.get("base__max_depth") is None:
            max_depth_values.extend([12, 20])
        else:
            max_depth_values.extend([max(6, params["base__max_depth"] - 6), params["base__max_depth"] + 6, None])

        leaf_values = sorted({1, params.get("base__min_samples_leaf", 1), params.get("base__min_samples_leaf", 1) + 1, max(1, params.get("base__min_samples_leaf", 1) - 1)})
        split_values = sorted({2, params.get("base__min_samples_split", 2), params.get("base__min_samples_split", 2) + 2, params.get("base__min_samples_split", 2) + 4})

        feature_values = []
        max_features = params.get("model__max_features")
        feature_values = sorted({round(max(0.15, max_features - 0.10), 2), round(max_features, 2), round(min(1.0, max_features + 0.10), 2)})

        max_samples = params.get("model__max_samples")
        sample_values = sorted({round(max(0.40, max_samples - 0.10), 2), round(max_samples, 2), round(min(1.0, max_samples + 0.10), 2)})

        criteria_values = [params.get("base__criterion")] + [criterion for criterion in TREE_CRITERIA if criterion != params.get("base__criterion")]
        bootstrap_feature_values = [params.get("model__bootstrap_features"), not params.get("model__bootstrap_features")]

        next_index = 1
        for value in max_depth_values[1:]:
            updated = {**params, "base__max_depth": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_depth_probe"))
            next_index += 1

        for value in leaf_values:
            if value == params.get("base__min_samples_leaf"):
                continue
            updated = {**params, "base__min_samples_leaf": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_leaf_probe"))
            next_index += 1

        for value in split_values:
            if value == params.get("base__min_samples_split"):
                continue
            updated = {**params, "base__min_samples_split": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_split_probe"))
            next_index += 1

        for value in feature_values:
            if value == params.get("model__max_features"):
                continue
            updated = {**params, "model__max_features": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_feature_probe"))
            next_index += 1

        for value in sample_values:
            if value == params.get("model__max_samples"):
                continue
            updated = {**params, "model__max_samples": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_sample_probe"))
            next_index += 1

        for value in criteria_values:
            if value == params.get("base__criterion"):
                continue
            updated = {**params, "base__criterion": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_criterion_probe"))
            next_index += 1

        for value in bootstrap_feature_values:
            if value == params.get("model__bootstrap_features"):
                continue
            updated = {**params, "model__bootstrap_features": value}
            candidates.append(candidate_record(f"stage4_{next_index:03d}", updated, "stage4_bootstrap_feature_probe"))
            next_index += 1

        candidates = dedupe_candidates(candidates)
        write_json_atomic(stage4_candidates_path, candidates)
        return candidates


    stage4_candidates = build_stage4_candidates()
    stage4_cv = StratifiedKFold(n_splits=PROFILE["stage4_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage4_splits = list(stage4_cv.split(X_train, y_train))


    def refresh_stage4_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage4_fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in stage4_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < PROFILE["stage4_cv"]:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": "stage4_local_cv",
                "n_estimators": best_tree_count,
                "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage4_summary_path)
        return summary_df


    stage4_summary = refresh_stage4_summary()
    completed_stage4 = set(stage4_summary["candidate_id"]) if not stage4_summary.empty else set()

    print("Stage 4 local candidates:", len(stage4_candidates))
    print("Stage 4 already completed:", len(completed_stage4))
    print("Best Stage 3 tree count carried into Stage 4:", best_tree_count)

    if RUN_STAGE_4:
        iterable = [candidate for candidate in stage4_candidates if candidate["candidate_id"] not in completed_stage4]
        for candidate in tqdm(iterable, desc="Stage 4 local refinement"):
            fold_df = read_dataframe(stage4_fold_path)
            done_folds = set(
                fold_df.loc[fold_df["candidate_id"] == candidate["candidate_id"], "fold_idx"].astype(int).tolist()
            ) if not fold_df.empty else set()

            for fold_idx, (train_idx, test_idx) in enumerate(stage4_splits):
                if fold_idx in done_folds:
                    continue

                start_time = time.time()
                model = build_bagging(
                    candidate["params"],
                    n_estimators=best_tree_count,
                    random_state=RANDOM_STATE + 100 + fold_idx,
                )
                model.fit(X_train[train_idx], y_train[train_idx])
                fold_pred = model.predict(X_train[test_idx])
                fold_accuracy = accuracy_score(y_train[test_idx], fold_pred)
                elapsed = time.time() - start_time

                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage4_local_cv",
                    "fold_idx": fold_idx,
                    "n_estimators": best_tree_count,
                    "fold_accuracy": float(fold_accuracy),
                    "fit_seconds": round(elapsed, 3),
                }
                append_rows(stage4_fold_path, [row], sort_cols=["candidate_id", "fold_idx"])
                del model
                gc.collect()

            stage4_summary = refresh_stage4_summary()
            if not stage4_summary.empty:
                update_manifest(
                    {
                        "stage4_completed_candidates": int(stage4_summary["candidate_id"].nunique()),
                        "stage4_total_candidates": len(stage4_candidates),
                        "stage4_best_cv_accuracy": float(stage4_summary["cv_mean_accuracy"].max()),
                    }
                )
                if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
                    sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG])

    stage4_summary = read_dataframe(stage4_summary_path)
    if not stage4_summary.empty:
        display(stage4_summary.head(12))
        checkpoint_housekeeping("stage4_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


PLOTS = dedent(
    """
    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        top_stage1 = stage1_results.sort_values("oob_accuracy", ascending=False).head(12).copy()
        top_stage1["label"] = top_stage1["candidate_id"] + " | " + top_stage1["base__criterion"].astype(str)
        sns.barplot(data=top_stage1, x="oob_accuracy", y="label", palette="crest")
        plt.title("Top Stage 1 bagging candidates by OOB accuracy")
        plt.xlabel("OOB accuracy")
        plt.ylabel("Candidate")
        path = save_current_figure("stage1_oob_top_candidates.png")
        plt.show()
        print("Saved figure:", path)

    if not stage2_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage2 = stage2_summary.sort_values("cv_mean_accuracy", ascending=False).head(12).copy()
        top_stage2["label"] = top_stage2["candidate_id"] + " | " + top_stage2["base__criterion"].astype(str)
        sns.barplot(data=top_stage2, x="cv_mean_accuracy", y="label", palette="mako")
        plt.title("Top Stage 2 bagging candidates by CV accuracy")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        path = save_current_figure("stage2_cv_top_candidates.png")
        plt.show()
        print("Saved figure:", path)

    if not stage3_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage3 = stage3_summary.sort_values("cv_mean_accuracy", ascending=False).drop_duplicates("candidate_id").head(5)
        selected_ids = top_stage3["candidate_id"].tolist()
        plot_df = stage3_summary[stage3_summary["candidate_id"].isin(selected_ids)].copy()
        sns.lineplot(data=plot_df, x="n_estimators", y="cv_mean_accuracy", hue="candidate_id", marker="o")
        plt.title("Stage 3 ensemble-growth curves")
        plt.xlabel("Number of estimators")
        plt.ylabel("Mean CV accuracy")
        path = save_current_figure("stage3_ensemble_growth_curves.png")
        plt.show()
        print("Saved figure:", path)

    if 'stage4_summary' in globals() and not stage4_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage4 = stage4_summary.sort_values("cv_mean_accuracy", ascending=False).head(12).copy()
        top_stage4["label"] = top_stage4["candidate_id"] + " | " + top_stage4["source"].astype(str)
        sns.barplot(data=top_stage4, x="cv_mean_accuracy", y="label", palette="rocket")
        plt.title("Top Stage 4 bagging local refinements")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        path = save_current_figure("stage4_local_refinement.png")
        plt.show()
        print("Saved figure:", path)
    """
).strip()


FINAL_MODEL = dedent(
    """
    if 'stage4_summary' in globals() and not stage4_summary.empty:
        final_search_table = stage4_summary.copy()
        final_stage_name = "stage4_local_cv"
    elif not stage3_summary.empty:
        final_search_table = stage3_summary.copy()
        final_stage_name = "stage3_ensemble_growth"
    elif not stage2_summary.empty:
        final_search_table = stage2_summary.copy()
        final_stage_name = "stage2_cv"
    else:
        raise RuntimeError("No completed search stage is available to train the final model.")

    best_row = final_search_table.sort_values(
        ["cv_mean_accuracy", "total_fit_seconds"] if "total_fit_seconds" in final_search_table.columns else ["cv_mean_accuracy"],
        ascending=[False, True] if "total_fit_seconds" in final_search_table.columns else [False],
    ).iloc[0]

    best_params = json.loads(best_row["params_json"])
    best_n_estimators = int(best_row["n_estimators"])

    print("Final stage used:", final_stage_name)
    print("Best params:", best_params)
    print("Best n_estimators:", best_n_estimators)

    if TRAIN_FINAL_MODEL:
        final_model = build_bagging(
            best_params,
            n_estimators=best_n_estimators,
            random_state=RANDOM_STATE + 999,
        )
        final_model.fit(X_train, y_train)
        valid_pred = final_model.predict(X_valid)
        validation_report = evaluate_predictions(y_valid, valid_pred)

        print("Validation accuracy:", round(validation_report["accuracy"], 4))
        print("Confusion matrix:")
        print(np.array(validation_report["confusion_matrix"]))

        disp = ConfusionMatrixDisplay.from_predictions(
            y_valid,
            valid_pred,
            display_labels=["Undamaged (0)", "Damaged (1)"],
            cmap="Blues",
            colorbar=False,
        )
        disp.ax_.set_title("Validation confusion matrix")
        confusion_path = save_current_figure("validation_confusion_matrix.png")
        plt.show()
        print("Saved figure:", confusion_path)

        global_importances = np.zeros(len(feature_names), dtype=np.float64)
        estimator_count = 0
        for estimator, feature_idx in zip(final_model.estimators_, final_model.estimators_features_):
            if not hasattr(estimator, "feature_importances_"):
                continue
            feature_idx = np.asarray(feature_idx, dtype=int)
            local_importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
            global_importances[feature_idx] += local_importance
            estimator_count += 1

        if estimator_count > 0:
            global_importances = global_importances / estimator_count
            importances = pd.Series(global_importances, index=feature_names).sort_values(ascending=False).head(20)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis")
            plt.title("Top 20 aggregated feature importances from final bagging")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            importance_path = save_current_figure("feature_importance_top20.png")
            plt.show()
            print("Saved figure:", importance_path)

        kaggle_model = build_bagging(
            best_params,
            n_estimators=best_n_estimators,
            random_state=RANDOM_STATE + 1001,
        )
        kaggle_model.fit(X_full, y_full)
        kaggle_pred = kaggle_model.predict(X_test_full)

        submission_df = pd.DataFrame(
            {
                "id": test_df["id"],
                "class": kaggle_pred.astype(int),
            }
        )
        submission_path = SUBMISSION_DIR / f"{NOTEBOOK_SLUG}_submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print("Submission saved to:", submission_path)

        final_summary = {
            "model_name": "Bagging over decision trees",
            "notebook_slug": NOTEBOOK_SLUG,
            "strategy": "colab_ultra_staged_resume_search",
            "search_profile": SEARCH_PROFILE,
            "bagging_n_jobs": BAGGING_N_JOBS,
            "best_stage": final_stage_name,
            "best_n_estimators": best_n_estimators,
            "best_params": best_params,
            "best_cv_accuracy": float(best_row["cv_mean_accuracy"]),
            "validation_accuracy": float(validation_report["accuracy"]),
            "submission_path": str(submission_path),
            "workspace_root": str(WORKSPACE_ROOT),
            "persist_root": str(PERSIST_ROOT),
        }

        summary_path = PERSIST_ROOT / "summary.json"
        write_json_atomic(summary_path, final_summary)
        checkpoint_housekeeping("final_model_complete", refresh_bundle=True, include_data_in_bundle=False)
        if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
            sync_workspace_to_drive()
        print("Summary saved to:", summary_path)
        final_summary
    """
).strip()


EXPORT_BUNDLE = dedent(
    """
    INCLUDE_DATA_IN_MANUAL_BUNDLE = True
    DOWNLOAD_BUNDLE_NOW = False

    bundle_path = create_resume_bundle(include_data=INCLUDE_DATA_IN_MANUAL_BUNDLE)
    print("Resume bundle saved to:", bundle_path)

    if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
        sync_workspace_to_drive([Path("exports")])
        print("Resume bundle also copied to Drive.")

    download_if_requested(bundle_path, should_download=DOWNLOAD_BUNDLE_NOW)
    """
).strip()


OUTRO = dedent(
    """
    ## Resume behavior

    If the Colab runtime disconnects or you stop execution:

    1. Reconnect the runtime.
    2. Restore your resume bundle, or enable Drive persistence before the next run.
    3. Run the notebook again from the top.
    4. Completed candidates will be skipped because checkpoints are read from the saved CSV and JSON files.

    The checkpoint directory is:

    `output/challenge_07_bagging_colab_ultra/checkpoints`
    """
).strip()


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown_cell(INTRO),
        markdown_cell("## 0. Imports and global constants"),
        code_cell(IMPORTS),
        markdown_cell("## 1. Create the Colab workspace"),
        code_cell(COLAB_SETUP),
        markdown_cell("## 2. Inspect the empty workspace and expected files"),
        code_cell(WORKSPACE_STATUS),
        markdown_cell("## 3. Optional: upload the three CSV files manually from the browser"),
        code_cell(UPLOAD_DATA),
        markdown_cell("## 4. Optional: restore a previous resume ZIP"),
        code_cell(UPLOAD_RESUME_BUNDLE),
        markdown_cell("## 5. Checkpoint helpers and reusable utilities"),
        code_cell(HELPERS),
        markdown_cell("## 6. Optional: enable Google Drive persistence"),
        code_cell(OPTIONAL_DRIVE),
        markdown_cell("## 7. Validate that the required CSV files are present"),
        code_cell(VALIDATE_DATA),
        markdown_cell("## 8. Runtime inspection and search budget"),
        code_cell(RUNTIME_CONFIG),
        markdown_cell("## 9. Load the challenge data"),
        code_cell(LOAD_DATA),
        markdown_cell("## 10. Stage 1: broad OOB screening"),
        code_cell(STAGE1),
        markdown_cell("## 11. Stage 2: full-train shortlist with 3-fold CV"),
        code_cell(STAGE2),
        markdown_cell("## 12. Stage 3: ensemble-growth evaluation on finalists"),
        code_cell(STAGE3),
        markdown_cell("## 13. Stage 4: narrow local refinement"),
        code_cell(STAGE4),
        markdown_cell("## 14. Diagnostic plots"),
        code_cell(PLOTS),
        markdown_cell("## 15. Final validation model and submission"),
        code_cell(FINAL_MODEL),
        markdown_cell("## 16. Optional: export and download a manual resume bundle"),
        code_cell(EXPORT_BUNDLE),
        markdown_cell(OUTRO),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return notebook


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Created {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
