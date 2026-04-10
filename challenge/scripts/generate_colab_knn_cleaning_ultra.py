from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "colab_knn_ultra"
NOTEBOOK_PATH = TARGET_DIR / "Challenge_01_KNN_Colab_Ultra.ipynb"


INTRO = dedent(
    """
    # Challenge 01 Colab Ultra: K-nearest Neighbors

    ## Objective

    This notebook is a Google Colab oriented replacement for the previous KNN searches.
    It is designed to be:

    1. Better aligned with the strongest model family found so far.
    2. Safe to start from a blank Colab session where only this notebook has been uploaded.
    3. Re-runnable with checkpoints so you do not lose the whole search after a disconnect.

    ## Search strategy

    The search is intentionally divided into four stages:

    1. Stage 1 performs a broad candidate screening on a reduced holdout split.
    2. Stage 2 promotes the shortlist to 3-fold cross-validation.
    3. Stage 3 performs a local 5-fold refinement around the strongest Stage 2 seeds.
    4. Stage 4 breaks ties using repeated holdout stability checks on the best refined candidates.

    ## Colab-specific design decisions

    - CPU runtime, not GPU. Standard `scikit-learn` KNN does not benefit meaningfully from default Colab GPU runtimes.
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
    from sklearn.decomposition import PCA
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

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
    NOTEBOOK_SLUG = "challenge_01_knn_colab_ultra"
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
        WORKSPACE_ROOT = Path("/content/challenge_knn_ultra_workspace")
    else:
        cwd = Path.cwd().resolve()
        if (cwd / "challenge" / "data" / "training.csv").exists():
            WORKSPACE_ROOT = cwd / "challenge"
        elif (cwd / "data" / "training.csv").exists():
            WORKSPACE_ROOT = cwd
        else:
            WORKSPACE_ROOT = cwd / "challenge_knn_ultra_workspace"

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
    DRIVE_PERSIST_ROOT = Path("/content/drive/MyDrive/challenge_knn_ultra_workspace")

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
    KNN_N_JOBS = max(1, CPU_COUNT - 1)

    SEARCH_PRESETS = {
        "balanced": {
            "stage1_n_iter": 80,
            "stage1_pool_fraction": 0.70,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 18,
            "stage2_cv": 3,
            "stage3_seed_top_k": 2,
            "stage3_cv": 5,
            "stage4_top_k": 4,
            "stage4_repeats": 4,
            "stage4_eval_size": 0.25,
        },
        "aggressive": {
            "stage1_n_iter": 120,
            "stage1_pool_fraction": 0.85,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 24,
            "stage2_cv": 3,
            "stage3_seed_top_k": 3,
            "stage3_cv": 5,
            "stage4_top_k": 5,
            "stage4_repeats": 6,
            "stage4_eval_size": 0.25,
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
            "knn_n_jobs": KNN_N_JOBS,
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

    if PROFILE["stage1_pool_fraction"] < 1.0:
        X_stage1_pool, _, y_stage1_pool, _ = train_test_split(
            X_train,
            y_train,
            train_size=PROFILE["stage1_pool_fraction"],
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_stage1_pool, y_stage1_pool = X_train, y_train

    X_stage1_fit, X_stage1_eval, y_stage1_fit, y_stage1_eval = train_test_split(
        X_stage1_pool,
        y_stage1_pool,
        test_size=PROFILE["stage1_eval_size"],
        stratify=y_stage1_pool,
        random_state=RANDOM_STATE,
    )

    print("Train split:", X_train.shape, y_train.shape)
    print("Validation split:", X_valid.shape, y_valid.shape)
    print("Stage 1 fit split:", X_stage1_fit.shape, y_stage1_fit.shape)
    print("Stage 1 eval split:", X_stage1_eval.shape, y_stage1_eval.shape)
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
        normalized["pca__n_components"] = int(normalized["pca__n_components"])
        normalized["model__n_neighbors"] = int(normalized["model__n_neighbors"])
        normalized["model__weights"] = str(normalized["model__weights"])
        normalized["model__metric"] = str(normalized["model__metric"])
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
            "pca__n_components": params.get("pca__n_components"),
            "model__n_neighbors": params.get("model__n_neighbors"),
            "model__weights": params.get("model__weights"),
            "model__metric": params.get("model__metric"),
            "params_json": json.dumps(params, sort_keys=True),
        }


    def build_knn(params: dict) -> KNeighborsClassifier:
        params = normalize_candidate(params)
        return KNeighborsClassifier(
            n_neighbors=int(params["model__n_neighbors"]),
            weights=params["model__weights"],
            metric=params["model__metric"],
            algorithm="brute",
            n_jobs=KNN_N_JOBS,
        )


    def fit_projection(X_train_input: np.ndarray, X_eval_input: np.ndarray, max_components: int) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_input)
        X_eval_scaled = scaler.transform(X_eval_input)
        safe_components = min(int(max_components), X_train_scaled.shape[0], X_train_scaled.shape[1])
        pca = PCA(n_components=safe_components, random_state=RANDOM_STATE)
        X_train_proj = pca.fit_transform(X_train_scaled).astype(np.float32)
        X_eval_proj = pca.transform(X_eval_scaled).astype(np.float32)
        return X_train_proj, X_eval_proj


    def evaluate_candidate(
        X_train_proj: np.ndarray,
        X_eval_proj: np.ndarray,
        y_train_input: np.ndarray,
        y_eval_input: np.ndarray,
        candidate: dict,
    ) -> tuple[float, float]:
        params = candidate["params"]
        n_components = int(params["pca__n_components"])
        model = build_knn(params)
        start_time = time.time()
        model.fit(X_train_proj[:, :n_components], y_train_input)
        predictions = model.predict(X_eval_proj[:, :n_components])
        elapsed = time.time() - start_time
        accuracy = accuracy_score(y_eval_input, predictions)
        return float(accuracy), round(elapsed, 3)


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


    def append_rows(path: Path, rows: list[dict], sort_cols: list[str] | None = None, ascending: bool | list[bool] = True) -> pd.DataFrame:
        new_df = pd.DataFrame(rows)
        current = read_dataframe(path)
        if current.empty:
            combined = new_df
        else:
            combined = pd.concat([current, new_df], ignore_index=True)
        if sort_cols:
            combined = combined.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        save_dataframe_atomic(combined, path)
        return combined


    def update_manifest(extra_payload: dict) -> dict:
        manifest_path = CHECKPOINT_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update(extra_payload)
        write_json_atomic(manifest_path, manifest)
        return manifest


    def create_resume_bundle(bundle_name: str = "challenge_01_knn_colab_ultra_resume.zip", include_data: bool = True) -> Path:
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


    def fit_knn_bundle(X_fit: np.ndarray, y_fit: np.ndarray, params: dict) -> dict:
        params = normalize_candidate(params)
        scaler = StandardScaler()
        X_fit_scaled = scaler.fit_transform(X_fit)
        pca = PCA(n_components=int(params["pca__n_components"]), random_state=RANDOM_STATE)
        X_fit_proj = pca.fit_transform(X_fit_scaled).astype(np.float32)
        model = build_knn(params)
        model.fit(X_fit_proj, y_fit)
        return {"scaler": scaler, "pca": pca, "model": model, "params": params}


    def predict_knn_bundle(bundle: dict, X_input: np.ndarray) -> np.ndarray:
        X_scaled = bundle["scaler"].transform(X_input)
        X_proj = bundle["pca"].transform(X_scaled).astype(np.float32)
        return bundle["model"].predict(X_proj)
    """
).strip()


STAGE1 = dedent(
    """
    STAGE1_SPACE = {
        "pca__n_components": [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72],
        "model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["manhattan", "euclidean", "chebyshev"],
    }

    stage1_candidates_path = CHECKPOINT_DIR / "stage1_candidates.json"
    stage1_results_path = CHECKPOINT_DIR / "stage1_holdout_results.csv"


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
            candidates.append(candidate_record(candidate_id, normalized, "stage1_random_holdout"))

        write_json_atomic(stage1_candidates_path, candidates)
        return candidates


    stage1_candidates = get_or_create_stage1_candidates()
    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    print("Stage 1 candidates:", len(stage1_candidates))
    print("Stage 1 already completed:", len(completed_stage1))

    if RUN_STAGE_1:
        max_components_stage1 = max(candidate["params"]["pca__n_components"] for candidate in stage1_candidates)
        X_stage1_fit_proj, X_stage1_eval_proj = fit_projection(X_stage1_fit, X_stage1_eval, max_components_stage1)

        iterable = [candidate for candidate in stage1_candidates if candidate["candidate_id"] not in completed_stage1]
        for candidate in tqdm(iterable, desc="Stage 1 holdout screening"):
            accuracy, elapsed = evaluate_candidate(
                X_stage1_fit_proj,
                X_stage1_eval_proj,
                y_stage1_fit,
                y_stage1_eval,
                candidate,
            )
            row = {
                **candidate_to_row(candidate),
                "stage": "stage1_holdout",
                "holdout_accuracy": accuracy,
                "fit_seconds": elapsed,
            }
            stage1_results = append_rows(
                stage1_results_path,
                [row],
                sort_cols=["holdout_accuracy", "fit_seconds"],
                ascending=[False, True],
            )

            update_manifest(
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_holdout_accuracy": float(stage1_results["holdout_accuracy"].max()),
                }
            )
            if ENABLE_GOOGLE_DRIVE_PERSISTENCE:
                sync_workspace_to_drive([Path("output") / NOTEBOOK_SLUG])

    stage1_results = read_dataframe(stage1_results_path)
    if not stage1_results.empty:
        stage1_results = stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True]).reset_index(drop=True)
        display(stage1_results.head(12))
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
        stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True])
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
    fold_df = read_dataframe(stage2_fold_path)
    completed_pairs = set(
        zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))
    ) if not fold_df.empty else set()

    print("Stage 2 shortlisted candidates:", len(stage2_candidates))
    print("Stage 2 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_2:
        max_components_stage2 = max(candidate["params"]["pca__n_components"] for candidate in stage2_candidates)

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(stage2_splits, desc="Stage 2 folds")):
            pending_candidates = [candidate for candidate in stage2_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            if not pending_candidates:
                continue

            X_fold_train_proj, X_fold_test_proj = fit_projection(
                X_train[train_idx],
                X_train[test_idx],
                max_components_stage2,
            )

            for candidate in tqdm(pending_candidates, desc=f"Stage 2 fold {fold_idx}", leave=False):
                accuracy, elapsed = evaluate_candidate(
                    X_fold_train_proj,
                    X_fold_test_proj,
                    y_train[train_idx],
                    y_train[test_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                }
                append_rows(stage2_fold_path, [row], sort_cols=["candidate_id", "fold_idx"])
                completed_pairs.add((candidate["candidate_id"], fold_idx))

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
        display(stage2_summary.head(12))
        checkpoint_housekeeping("stage2_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STAGE3 = dedent(
    """
    stage3_candidates_path = CHECKPOINT_DIR / "stage3_local_candidates.json"
    stage3_fold_path = CHECKPOINT_DIR / "stage3_local_cv_fold_results.csv"
    stage3_summary_path = CHECKPOINT_DIR / "stage3_local_cv_summary.csv"

    if stage2_summary.empty:
        raise RuntimeError("Stage 2 produced no results. Run Stage 2 before Stage 3.")

    stage3_seed_rows = (
        stage2_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage3_seed_top_k"])
        .copy()
    )


    def build_stage3_candidates() -> list[dict]:
        if stage3_candidates_path.exists():
            return json.loads(stage3_candidates_path.read_text())

        candidates = []
        seen = set()
        next_index = 0

        for seed_rank, row in enumerate(stage3_seed_rows.to_dict(orient="records")):
            seed_params = json.loads(row["params_json"])
            best_components = int(seed_params["pca__n_components"])
            best_k = int(seed_params["model__n_neighbors"])
            best_metric = str(seed_params["model__metric"])
            best_weight = str(seed_params["model__weights"])

            component_values = sorted(
                {
                    value
                    for value in [
                        best_components - 6,
                        best_components - 4,
                        best_components - 2,
                        best_components - 1,
                        best_components,
                        best_components + 1,
                        best_components + 2,
                        best_components + 4,
                        best_components + 6,
                    ]
                    if 12 <= value <= X_train.shape[1]
                }
            )

            neighbor_values = sorted(
                {
                    value
                    for value in [
                        best_k - 3,
                        best_k - 2,
                        best_k - 1,
                        best_k,
                        best_k + 1,
                        best_k + 2,
                        best_k + 3,
                        best_k + 5,
                    ]
                    if value >= 1
                }
            )

            weight_values = sorted({best_weight, "uniform", "distance"})
            metric_values = sorted({best_metric, "manhattan", "euclidean", "chebyshev"})

            for components in component_values:
                for neighbors in neighbor_values:
                    for weight in weight_values:
                        for metric in metric_values:
                            params = normalize_candidate(
                                {
                                    "pca__n_components": components,
                                    "model__n_neighbors": neighbors,
                                    "model__weights": weight,
                                    "model__metric": metric,
                                }
                            )
                            signature = candidate_signature(params)
                            if signature in seen:
                                continue
                            seen.add(signature)
                            candidate_id = f"stage3_{next_index:03d}"
                            next_index += 1
                            candidates.append(candidate_record(candidate_id, params, f"stage3_seed_{seed_rank}"))

        write_json_atomic(stage3_candidates_path, candidates)
        return candidates


    stage3_candidates = build_stage3_candidates()
    stage3_cv = StratifiedKFold(n_splits=PROFILE["stage3_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage3_splits = list(stage3_cv.split(X_train, y_train))


    def refresh_stage3_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage3_fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in stage3_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < PROFILE["stage3_cv"]:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": "stage3_local_cv",
                "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage3_summary_path)
        return summary_df


    stage3_summary = refresh_stage3_summary()
    fold_df = read_dataframe(stage3_fold_path)
    completed_pairs = set(
        zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))
    ) if not fold_df.empty else set()

    print("Stage 3 local candidates:", len(stage3_candidates))
    print("Stage 3 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_3:
        max_components_stage3 = max(candidate["params"]["pca__n_components"] for candidate in stage3_candidates)

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(stage3_splits, desc="Stage 3 folds")):
            pending_candidates = [candidate for candidate in stage3_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            if not pending_candidates:
                continue

            X_fold_train_proj, X_fold_test_proj = fit_projection(
                X_train[train_idx],
                X_train[test_idx],
                max_components_stage3,
            )

            for candidate in tqdm(pending_candidates, desc=f"Stage 3 fold {fold_idx}", leave=False):
                accuracy, elapsed = evaluate_candidate(
                    X_fold_train_proj,
                    X_fold_test_proj,
                    y_train[train_idx],
                    y_train[test_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage3_local_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                }
                append_rows(stage3_fold_path, [row], sort_cols=["candidate_id", "fold_idx"])
                completed_pairs.add((candidate["candidate_id"], fold_idx))

            stage3_summary = refresh_stage3_summary()
            if not stage3_summary.empty:
                update_manifest(
                    {
                        "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                        "stage3_total_candidates": len(stage3_candidates),
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
    stage4_split_path = CHECKPOINT_DIR / "stage4_stability_split_results.csv"
    stage4_summary_path = CHECKPOINT_DIR / "stage4_stability_summary.csv"

    if stage3_summary.empty:
        raise RuntimeError("Stage 3 produced no results. Run Stage 3 before Stage 4.")

    stage4_seed_rows = (
        stage3_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage4_top_k"])
        .copy()
    )

    stage4_candidates = []
    for row in stage4_seed_rows.to_dict(orient="records"):
        stage4_candidates.append(
            candidate_record(
                candidate_id=row["candidate_id"],
                params=json.loads(row["params_json"]),
                source="stage4_top_from_stage3",
            )
        )


    def refresh_stage4_summary() -> pd.DataFrame:
        split_df = read_dataframe(stage4_split_path)
        if split_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in stage4_candidates:
            candidate_split_df = split_df[split_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_split_df) < PROFILE["stage4_repeats"]:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": "stage4_stability",
                "mean_accuracy": float(candidate_split_df["split_accuracy"].mean()),
                "std_accuracy": float(candidate_split_df["split_accuracy"].std(ddof=0)),
                "total_fit_seconds": float(candidate_split_df["fit_seconds"].sum()),
                "completed_splits": int(candidate_split_df["split_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["mean_accuracy", "std_accuracy", "total_fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage4_summary_path)
        return summary_df


    stage4_summary = refresh_stage4_summary()
    split_df = read_dataframe(stage4_split_path)
    completed_pairs = set(
        zip(split_df["candidate_id"].astype(str), split_df["split_idx"].astype(int))
    ) if not split_df.empty else set()

    print("Stage 4 stability candidates:", len(stage4_candidates))
    print("Stage 4 completed split-pairs:", len(completed_pairs))

    if RUN_STAGE_4:
        max_components_stage4 = max(candidate["params"]["pca__n_components"] for candidate in stage4_candidates)

        for split_idx in tqdm(range(PROFILE["stage4_repeats"]), desc="Stage 4 stability splits"):
            pending_candidates = [candidate for candidate in stage4_candidates if (candidate["candidate_id"], split_idx) not in completed_pairs]
            if not pending_candidates:
                continue

            X_split_train, X_split_eval, y_split_train, y_split_eval = train_test_split(
                X_train,
                y_train,
                test_size=PROFILE["stage4_eval_size"],
                stratify=y_train,
                random_state=RANDOM_STATE + 500 + split_idx,
            )

            X_split_train_proj, X_split_eval_proj = fit_projection(
                X_split_train,
                X_split_eval,
                max_components_stage4,
            )

            for candidate in tqdm(pending_candidates, desc=f"Stage 4 split {split_idx}", leave=False):
                accuracy, elapsed = evaluate_candidate(
                    X_split_train_proj,
                    X_split_eval_proj,
                    y_split_train,
                    y_split_eval,
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage4_stability",
                    "split_idx": split_idx,
                    "split_accuracy": accuracy,
                    "fit_seconds": elapsed,
                }
                append_rows(stage4_split_path, [row], sort_cols=["candidate_id", "split_idx"])
                completed_pairs.add((candidate["candidate_id"], split_idx))

            stage4_summary = refresh_stage4_summary()
            if not stage4_summary.empty:
                update_manifest(
                    {
                        "stage4_completed_candidates": int(stage4_summary["candidate_id"].nunique()),
                        "stage4_total_candidates": len(stage4_candidates),
                        "stage4_best_mean_accuracy": float(stage4_summary["mean_accuracy"].max()),
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
        top_stage1 = stage1_results.sort_values("holdout_accuracy", ascending=False).head(12).copy()
        top_stage1["label"] = (
            top_stage1["candidate_id"]
            + " | c="
            + top_stage1["pca__n_components"].astype(int).astype(str)
            + " | k="
            + top_stage1["model__n_neighbors"].astype(int).astype(str)
        )
        sns.barplot(data=top_stage1, x="holdout_accuracy", y="label", palette="crest")
        plt.title("Top Stage 1 KNN candidates by holdout accuracy")
        plt.xlabel("Holdout accuracy")
        plt.ylabel("Candidate")
        path = save_current_figure("stage1_top_candidates.png")
        plt.show()
        print("Saved figure:", path)

    if not stage2_summary.empty:
        stage2_plot = stage2_summary.copy()
        best_metric = stage2_plot.iloc[0]["model__metric"]
        best_weight = stage2_plot.iloc[0]["model__weights"]
        heatmap_data = (
            stage2_plot[
                (stage2_plot["model__metric"] == best_metric)
                & (stage2_plot["model__weights"] == best_weight)
            ]
            .pivot_table(
                index="pca__n_components",
                columns="model__n_neighbors",
                values="cv_mean_accuracy",
            )
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Stage 2 CV heatmap\\nmetric={best_metric}, weights={best_weight}")
        plt.xlabel("Number of neighbors")
        plt.ylabel("PCA components")
        path = save_current_figure("stage2_cv_heatmap.png")
        plt.show()
        print("Saved figure:", path)

    if 'stage3_summary' in globals() and not stage3_summary.empty:
        stage3_plot = stage3_summary.copy()
        best_metric = stage3_plot.iloc[0]["model__metric"]
        best_weight = stage3_plot.iloc[0]["model__weights"]
        heatmap_data = (
            stage3_plot[
                (stage3_plot["model__metric"] == best_metric)
                & (stage3_plot["model__weights"] == best_weight)
            ]
            .pivot_table(
                index="pca__n_components",
                columns="model__n_neighbors",
                values="cv_mean_accuracy",
            )
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="mako")
        plt.title(f"Stage 3 local refinement heatmap\\nmetric={best_metric}, weights={best_weight}")
        plt.xlabel("Number of neighbors")
        plt.ylabel("PCA components")
        path = save_current_figure("stage3_local_heatmap.png")
        plt.show()
        print("Saved figure:", path)

    if 'stage4_summary' in globals() and not stage4_summary.empty:
        top_stage4 = stage4_summary.head(10).copy()
        top_stage4["label"] = (
            top_stage4["candidate_id"]
            + " | c="
            + top_stage4["pca__n_components"].astype(int).astype(str)
            + " | k="
            + top_stage4["model__n_neighbors"].astype(int).astype(str)
        )
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_stage4, x="mean_accuracy", y="label", palette="rocket")
        plt.title("Top Stage 4 KNN candidates by stability mean accuracy")
        plt.xlabel("Mean repeated-holdout accuracy")
        plt.ylabel("Candidate")
        path = save_current_figure("stage4_stability_top_candidates.png")
        plt.show()
        print("Saved figure:", path)
    """
).strip()


FINAL_MODEL = dedent(
    """
    if 'stage4_summary' in globals() and not stage4_summary.empty:
        final_search_table = stage4_summary.copy()
        final_stage_name = "stage4_stability"
        best_row = final_search_table.sort_values(["mean_accuracy", "std_accuracy", "total_fit_seconds"], ascending=[False, True, True]).iloc[0]
        best_score = float(best_row["mean_accuracy"])
    elif 'stage3_summary' in globals() and not stage3_summary.empty:
        final_search_table = stage3_summary.copy()
        final_stage_name = "stage3_local_cv"
        best_row = final_search_table.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).iloc[0]
        best_score = float(best_row["cv_mean_accuracy"])
    elif not stage2_summary.empty:
        final_search_table = stage2_summary.copy()
        final_stage_name = "stage2_cv"
        best_row = final_search_table.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).iloc[0]
        best_score = float(best_row["cv_mean_accuracy"])
    else:
        final_search_table = stage1_results.copy()
        final_stage_name = "stage1_holdout"
        best_row = final_search_table.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True]).iloc[0]
        best_score = float(best_row["holdout_accuracy"])

    best_params = json.loads(best_row["params_json"])

    print("Final stage used:", final_stage_name)
    print("Best params:", best_params)
    print("Best search-stage score:", round(best_score, 4))

    if TRAIN_FINAL_MODEL:
        final_bundle = fit_knn_bundle(X_train, y_train, best_params)
        valid_pred = predict_knn_bundle(final_bundle, X_valid)
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

        full_bundle = fit_knn_bundle(X_full, y_full, best_params)
        kaggle_pred = predict_knn_bundle(full_bundle, X_test_full)

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
            "model_name": "K-nearest neighbors",
            "notebook_slug": NOTEBOOK_SLUG,
            "strategy": "colab_ultra_staged_resume_search",
            "search_profile": SEARCH_PROFILE,
            "knn_n_jobs": KNN_N_JOBS,
            "best_stage": final_stage_name,
            "best_params": best_params,
            "best_search_stage_score": best_score,
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

    `output/challenge_01_knn_colab_ultra/checkpoints`
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
        markdown_cell("## 10. Stage 1: broad holdout screening"),
        code_cell(STAGE1),
        markdown_cell("## 11. Stage 2: shortlist with 3-fold CV"),
        code_cell(STAGE2),
        markdown_cell("## 12. Stage 3: local 5-fold refinement"),
        code_cell(STAGE3),
        markdown_cell("## 13. Stage 4: repeated-holdout stability tie-break"),
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
