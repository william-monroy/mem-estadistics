from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent, indent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StrategySpec:
    folder_name: str
    notebook_name: str
    notebook_title: str
    notebook_slug: str
    workspace_name: str
    readme_title: str
    readme_text: str
    intro: str
    imports_code: str
    runtime_code: str
    strategy_helpers_code: str
    stage1_code: str
    stage2_code: str
    stage3_code: str
    plots_code: str
    final_code: str
    outro: str


REQUIREMENTS_TEXT = dedent(
    """
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    scipy
    tqdm
    """
).strip() + "\n"


WORKSPACE_TEMPLATE_TEXT = dedent(
    """
    Esta carpeta es solo una referencia de la estructura esperada del workspace cuando ejecutes la notebook en Google Colab.

    El flujo normal sigue siendo:

    1. Subir el notebook a Colab.
    2. Ejecutar las celdas iniciales.
    3. Subir `training.csv`, `test.csv` y `sample.csv`, o restaurar un ZIP de reanudacion.
    4. Descargar el ZIP exportado al final de cada etapa importante si quieres continuar luego en otra sesion.
    """
).strip() + "\n"


COMMON_UPLOAD_DATA = dedent(
    """
    UPLOAD_DATA_FILES = False

    if UPLOAD_DATA_FILES:
        if not IN_COLAB:
            raise RuntimeError("This upload helper is intended for Google Colab.")

        uploaded = files.upload()
        for original_name, file_bytes in uploaded.items():
            filename = Path(original_name).name
            target_path = DATA_DIR / filename
            target_path.write_bytes(file_bytes)
            print("Saved:", target_path)
    else:
        print("Set UPLOAD_DATA_FILES = True if you want to upload training.csv, test.csv and sample.csv.")
    """
).strip()


COMMON_VALIDATE_DATA = dedent(
    """
    missing_files = [str(path) for path in [TRAIN_PATH, TEST_PATH, SAMPLE_PATH] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required data files. Upload training.csv, test.csv and sample.csv first or restore a resume ZIP. "
            f"Missing: {missing_files}"
        )

    print("All required CSV files are present.")
    """
).strip()


COMMON_LOAD_DATA = dedent(
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_df = pd.read_csv(SAMPLE_PATH)

    feature_names = [column for column in train_df.columns if column not in {"id", "class"}]
    train_ids = train_df["id"].to_numpy()
    test_ids = test_df["id"].to_numpy()

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

    print("Train split:", X_train.shape, y_train.shape)
    print("Validation split:", X_valid.shape, y_valid.shape)
    """
).strip()


COMMON_EXPORT_BUNDLE = dedent(
    """
    INCLUDE_DATA_IN_MANUAL_BUNDLE = True
    DOWNLOAD_BUNDLE_NOW = False

    bundle_path = create_resume_bundle(include_data=INCLUDE_DATA_IN_MANUAL_BUNDLE)
    print("Resume bundle saved to:", bundle_path)

    if DOWNLOAD_BUNDLE_NOW and IN_COLAB:
        files.download(str(bundle_path))
    """
).strip()


def common_imports(notebook_slug: str, extra_imports: str) -> str:
    extra_imports_block = indent(extra_imports.strip(), "        ")
    return dedent(
        f"""
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        import gc
        import itertools
        import json
        import platform
        import time
        import warnings
        import zipfile
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import sklearn
        from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
        from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
{extra_imports_block}

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
        NOTEBOOK_SLUG = "{notebook_slug}"
        """
    ).strip()


def common_setup(workspace_name: str, allow_multi_zip: bool = False) -> tuple[str, str, str]:
    setup_code = dedent(
        f"""
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
            WORKSPACE_ROOT = Path("/content/{workspace_name}")
        else:
            cwd = Path.cwd().resolve()
            if (cwd / "challenge" / "data" / "training.csv").exists():
                WORKSPACE_ROOT = cwd / "challenge"
            elif (cwd / "data" / "training.csv").exists():
                WORKSPACE_ROOT = cwd
            else:
                WORKSPACE_ROOT = cwd / "{workspace_name}"

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

        print("IN_COLAB:", IN_COLAB)
        print("WORKSPACE_ROOT:", WORKSPACE_ROOT)
        print("PERSIST_ROOT:", PERSIST_ROOT)
        """
    ).strip()

    status_code = dedent(
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
        """
    ).strip()

    if allow_multi_zip:
        restore_code = dedent(
            """
            RESTORE_RESUME_BUNDLES = False

            if RESTORE_RESUME_BUNDLES:
                if not IN_COLAB:
                    raise RuntimeError("This restore helper is intended for Google Colab.")

                uploaded = files.upload()
                zip_names = [Path(name).name for name in uploaded if str(name).lower().endswith(".zip")]
                if not zip_names:
                    raise ValueError("Upload at least one ZIP file when restoring artifacts.")

                for bundle_name in zip_names:
                    bundle_path = EXPORT_DIR / bundle_name
                    bundle_path.write_bytes(uploaded[bundle_name])
                    with zipfile.ZipFile(bundle_path, "r") as zip_file:
                        zip_file.extractall(WORKSPACE_ROOT)
                    print("Restored:", bundle_path)
            else:
                print("Set RESTORE_RESUME_BUNDLES = True if you want to restore one or more ZIP bundles.")
            """
        ).strip()
    else:
        restore_code = dedent(
            """
            RESTORE_RESUME_BUNDLE = False

            if RESTORE_RESUME_BUNDLE:
                if not IN_COLAB:
                    raise RuntimeError("This restore helper is intended for Google Colab.")

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

    return setup_code, status_code, restore_code


def common_helpers(default_bundle_name: str) -> str:
    return dedent(
        f"""
        DEFAULT_BUNDLE_NAME = "{default_bundle_name}"


        def save_current_figure(filename: str) -> Path:
            path = PERSIST_ROOT / filename
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


        def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(path)


        def read_dataframe(path: Path) -> pd.DataFrame:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()


        def normalize_value(value):
            if isinstance(value, np.generic):
                return value.item()
            return value


        def candidate_signature(params: dict) -> str:
            normalized = {{key: normalize_value(value) for key, value in params.items()}}
            return json.dumps(normalized, sort_keys=True)


        def update_manifest(extra_payload: dict) -> dict:
            manifest_path = CHECKPOINT_DIR / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {{}}
            manifest.update(extra_payload)
            write_json_atomic(manifest_path, manifest)
            return manifest


        def create_resume_bundle(bundle_name: str = DEFAULT_BUNDLE_NAME, include_data: bool = True) -> Path:
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


        def checkpoint_housekeeping(stage_name: str, *, refresh_bundle: bool = False, include_data_in_bundle: bool = False) -> dict:
            payload = {{"last_checkpoint_stage": stage_name}}
            if refresh_bundle:
                bundle_path = create_resume_bundle(include_data=include_data_in_bundle)
                payload["resume_bundle_path"] = str(bundle_path)
                payload["resume_bundle_size_mb"] = round(bundle_path.stat().st_size / (1024 ** 2), 3)
            update_manifest(payload)
            return payload
        """
    ).strip()


KNN_CLEANING_INTRO = dedent(
    """
    # Challenge 08 Colab Ultra: KNN with instance cleaning

    Esta variante ataca el punto mas prometedor que vimos hasta ahora:

    - `KNN` sigue siendo una de las mejores familias del challenge.
    - La limpieza de instancias ruidosas mejora la validacion local.
    - La meta es explotar la regla que permite cualquier tecnica de preprocesamiento si queda bien documentada.

    ## Estrategia

    1. `Stage 1`: screening amplio en holdout para distintas estrategias de limpieza y parametros KNN.
    2. `Stage 2`: shortlist con `3-fold CV`.
    3. `Stage 3`: refinamiento local alrededor de los mejores seeds.
    4. Modelo final, submission y export de artefactos para stacking.

    ## Tecnicas de limpieza incluidas

    - `none`
    - `lof_0.03`
    - `lof_0.05`
    - `tomek`
    - `enn3`
    - `enn5`
    - `renn3`
    - `renn5`
    """
).strip()


KNN_CLEANING_IMPORTS = common_imports(
    "challenge_08_knn_cleaning_colab_ultra",
    dedent(
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
        from sklearn.preprocessing import RobustScaler, StandardScaler
        """
    ).strip(),
)


KNN_CLEANING_RUNTIME = dedent(
    """
    CPU_COUNT = os.cpu_count() or 2
    RAM_GB = None if psutil is None else round(psutil.virtual_memory().total / (1024 ** 3), 2)
    KNN_N_JOBS = max(1, CPU_COUNT - 1)

    SEARCH_PRESETS = {
        "balanced": {
            "stage1_n_iter": 72,
            "stage1_pool_fraction": 0.70,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 16,
            "stage2_cv": 3,
            "stage3_seed_top_k": 3,
            "stage3_cv": 5,
        },
        "aggressive": {
            "stage1_n_iter": 108,
            "stage1_pool_fraction": 0.82,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 20,
            "stage2_cv": 3,
            "stage3_seed_top_k": 4,
            "stage3_cv": 5,
        },
    }

    SEARCH_PROFILE = "aggressive"
    PROFILE = SEARCH_PRESETS[SEARCH_PROFILE]

    RUN_STAGE_1 = True
    RUN_STAGE_2 = True
    RUN_STAGE_3 = True
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


KNN_CLEANING_HELPERS = dedent(
    """
    CLEANING_METHODS = ["none", "lof_0.03", "lof_0.05", "tomek", "enn3", "enn5", "renn3", "renn5"]


    def normalize_candidate(params: dict) -> dict:
        normalized = {key: normalize_value(value) for key, value in params.items()}
        normalized["clean__method"] = str(normalized["clean__method"])
        normalized["scale__method"] = str(normalized["scale__method"])
        normalized["pca__n_components"] = int(normalized["pca__n_components"])
        normalized["model__n_neighbors"] = int(normalized["model__n_neighbors"])
        normalized["model__metric"] = str(normalized["model__metric"])
        normalized["model__weights"] = "distance"
        return normalized


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
            "clean__method": params["clean__method"],
            "scale__method": params["scale__method"],
            "pca__n_components": params["pca__n_components"],
            "model__n_neighbors": params["model__n_neighbors"],
            "model__metric": params["model__metric"],
            "params_json": json.dumps(params, sort_keys=True),
        }


    def choose_scaler(name: str):
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler: {name}")


    def _apply_tomek_links(X_input: np.ndarray, y_input: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        if len(X_input) < 3:
            return X_input, y_input, 0
        scaled = StandardScaler().fit_transform(X_input)
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(scaled)
        nearest = nn.kneighbors(scaled, return_distance=False)[:, 1]
        remove = set()
        for i, j in enumerate(nearest):
            if nearest[j] == i and y_input[i] != y_input[j]:
                remove.add(i)
                remove.add(int(j))
        if not remove:
            return X_input, y_input, 0
        keep_mask = np.ones(len(X_input), dtype=bool)
        keep_mask[list(remove)] = False
        return X_input[keep_mask], y_input[keep_mask], int((~keep_mask).sum())


    def _apply_enn_once(X_input: np.ndarray, y_input: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
        if len(X_input) <= k + 1:
            return X_input, y_input, 0
        scaled = StandardScaler().fit_transform(X_input)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_input)))
        nn.fit(scaled)
        neighbors = nn.kneighbors(scaled, return_distance=False)[:, 1:]
        keep_mask = np.ones(len(X_input), dtype=bool)
        for i, idx in enumerate(neighbors):
            votes = y_input[idx]
            pred = int(votes.mean() >= 0.5)
            if pred != y_input[i]:
                keep_mask[i] = False
        return X_input[keep_mask], y_input[keep_mask], int((~keep_mask).sum())


    def apply_cleaning(X_input: np.ndarray, y_input: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if method == "none":
            return X_input, y_input, {"removed_rows": 0}

        if method.startswith("lof_"):
            contamination = float(method.split("_")[1])
            scaled = StandardScaler().fit_transform(X_input)
            keep_mask = np.ones(len(X_input), dtype=bool)
            for cls in np.unique(y_input):
                idx = np.where(y_input == cls)[0]
                if len(idx) < 12:
                    continue
                n_neighbors = min(25, len(idx) - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                pred = lof.fit_predict(scaled[idx])
                keep_mask[idx] = pred == 1
            return X_input[keep_mask], y_input[keep_mask], {"removed_rows": int((~keep_mask).sum())}

        if method == "tomek":
            X_clean, y_clean, removed = _apply_tomek_links(X_input, y_input)
            return X_clean, y_clean, {"removed_rows": removed}

        if method in {"enn3", "enn5"}:
            k = int(method.replace("enn", ""))
            X_clean, y_clean, removed = _apply_enn_once(X_input, y_input, k)
            return X_clean, y_clean, {"removed_rows": removed}

        if method in {"renn3", "renn5"}:
            k = int(method.replace("renn", ""))
            X_curr, y_curr = X_input, y_input
            total_removed = 0
            for _ in range(3):
                X_next, y_next, removed = _apply_enn_once(X_curr, y_curr, k)
                total_removed += removed
                if removed == 0 or len(X_next) < 50:
                    X_curr, y_curr = X_next, y_next
                    break
                X_curr, y_curr = X_next, y_next
            return X_curr, y_curr, {"removed_rows": total_removed}

        raise ValueError(f"Unknown cleaning method: {method}")


    def build_knn(candidate: dict) -> KNeighborsClassifier:
        params = candidate["params"]
        return KNeighborsClassifier(
            n_neighbors=int(params["model__n_neighbors"]),
            weights="distance",
            metric=params["model__metric"],
            algorithm="brute",
            n_jobs=KNN_N_JOBS,
        )


    def fit_bundle(X_fit: np.ndarray, y_fit: np.ndarray, candidate: dict) -> dict:
        params = candidate["params"]
        X_clean, y_clean, cleaning_info = apply_cleaning(X_fit, y_fit, params["clean__method"])
        scaler = choose_scaler(params["scale__method"])
        X_clean_scaled = scaler.fit_transform(X_clean)
        n_components = min(int(params["pca__n_components"]), X_clean_scaled.shape[0], X_clean_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_clean_proj = pca.fit_transform(X_clean_scaled).astype(np.float32)
        model = build_knn(candidate)
        model.fit(X_clean_proj, y_clean)
        return {
            "candidate": candidate,
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "cleaning_info": cleaning_info,
        }


    def predict_bundle(bundle: dict, X_input: np.ndarray, proba: bool = False) -> np.ndarray:
        X_scaled = bundle["scaler"].transform(X_input)
        X_proj = bundle["pca"].transform(X_scaled).astype(np.float32)
        if proba:
            return bundle["model"].predict_proba(X_proj)
        return bundle["model"].predict(X_proj)


    def evaluate_candidate(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, candidate: dict) -> tuple[float, float, int]:
        start = time.time()
        bundle = fit_bundle(X_fit, y_fit, candidate)
        predictions = predict_bundle(bundle, X_eval, proba=False)
        accuracy = accuracy_score(y_eval, predictions)
        elapsed = round(time.time() - start, 3)
        removed_rows = int(bundle["cleaning_info"].get("removed_rows", 0))
        return float(accuracy), elapsed, removed_rows


    def refresh_summary_from_folds(fold_path: Path, summary_path: Path, candidates: list[dict], n_splits: int, stage_name: str) -> pd.DataFrame:
        fold_df = read_dataframe(fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if candidate_fold_df["fold_idx"].nunique() < n_splits:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": stage_name,
                "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                "mean_removed_rows": float(candidate_fold_df["removed_rows"].mean()),
                "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "mean_removed_rows", "total_fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, summary_path)
        return summary_df
    """
).strip()


KNN_CLEANING_STAGE1 = dedent(
    """
    STAGE1_SPACE = {
        "clean__method": CLEANING_METHODS,
        "scale__method": ["standard", "robust"],
        "pca__n_components": [32, 36, 40, 44, 48, 52, 56, 60, 64],
        "model__n_neighbors": [3, 4, 5, 6, 7, 9, 11],
        "model__metric": ["manhattan", "euclidean"],
    }

    stage1_candidates_path = CHECKPOINT_DIR / "stage1_candidates.json"
    stage1_results_path = CHECKPOINT_DIR / "stage1_holdout_results.csv"


    def get_or_create_stage1_candidates() -> list[dict]:
        if stage1_candidates_path.exists():
            return json.loads(stage1_candidates_path.read_text())

        X_stage1_pool, _, y_stage1_pool, _ = train_test_split(
            X_train,
            y_train,
            train_size=PROFILE["stage1_pool_fraction"],
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        globals()["X_stage1_pool"] = X_stage1_pool
        globals()["y_stage1_pool"] = y_stage1_pool

        sampler = ParameterSampler(STAGE1_SPACE, n_iter=PROFILE["stage1_n_iter"], random_state=RANDOM_STATE)
        candidates = []
        seen = set()
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


    X_stage1_pool, _, y_stage1_pool, _ = train_test_split(
        X_train,
        y_train,
        train_size=PROFILE["stage1_pool_fraction"],
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_stage1_fit, X_stage1_eval, y_stage1_fit, y_stage1_eval = train_test_split(
        X_stage1_pool,
        y_stage1_pool,
        test_size=PROFILE["stage1_eval_size"],
        stratify=y_stage1_pool,
        random_state=RANDOM_STATE,
    )

    stage1_candidates = get_or_create_stage1_candidates()
    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    print("Stage 1 candidates:", len(stage1_candidates))
    print("Stage 1 completed:", len(completed_stage1))

    if RUN_STAGE_1:
        for candidate in tqdm([c for c in stage1_candidates if c["candidate_id"] not in completed_stage1], desc="Stage 1 screening"):
            accuracy, elapsed, removed_rows = evaluate_candidate(X_stage1_fit, y_stage1_fit, X_stage1_eval, y_stage1_eval, candidate)
            row = {
                **candidate_to_row(candidate),
                "stage": "stage1_holdout",
                "holdout_accuracy": accuracy,
                "fit_seconds": elapsed,
                "removed_rows": removed_rows,
            }
            stage1_results = pd.concat([stage1_results, pd.DataFrame([row])], ignore_index=True) if not stage1_results.empty else pd.DataFrame([row])
            stage1_results = stage1_results.sort_values(["holdout_accuracy", "removed_rows", "fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(stage1_results, stage1_results_path)
            update_manifest(
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_holdout_accuracy": float(stage1_results["holdout_accuracy"].max()),
                }
            )
            gc.collect()

    stage1_results = read_dataframe(stage1_results_path)
    if not stage1_results.empty:
        display(stage1_results.head(12))
        checkpoint_housekeeping("stage1_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


KNN_CLEANING_STAGE2 = dedent(
    """
    stage2_fold_path = CHECKPOINT_DIR / "stage2_cv_fold_results.csv"
    stage2_summary_path = CHECKPOINT_DIR / "stage2_cv_summary.csv"

    if stage1_results.empty:
        raise RuntimeError("Stage 1 produced no results.")

    stage2_shortlist = (
        stage1_results.sort_values(["holdout_accuracy", "removed_rows", "fit_seconds"], ascending=[False, True, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage2_top_k"])
        .copy()
    )

    stage2_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage2_shortlist_from_stage1")
        for row in stage2_shortlist.to_dict(orient="records")
    ]

    stage2_cv = StratifiedKFold(n_splits=PROFILE["stage2_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage2_splits = list(stage2_cv.split(X_train, y_train))

    fold_df = read_dataframe(stage2_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()

    print("Stage 2 candidates:", len(stage2_candidates))
    print("Stage 2 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_2:
        for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(stage2_splits, desc="Stage 2 folds")):
            pending = [candidate for candidate in stage2_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            for candidate in tqdm(pending, desc=f"Stage 2 fold {fold_idx}", leave=False):
                accuracy, elapsed, removed_rows = evaluate_candidate(
                    X_train[fit_idx],
                    y_train[fit_idx],
                    X_train[eval_idx],
                    y_train[eval_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                    "removed_rows": removed_rows,
                }
                fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True) if not fold_df.empty else pd.DataFrame([row])
                save_dataframe_atomic(fold_df, stage2_fold_path)
                completed_pairs.add((candidate["candidate_id"], fold_idx))
                gc.collect()

            stage2_summary = refresh_summary_from_folds(stage2_fold_path, stage2_summary_path, stage2_candidates, PROFILE["stage2_cv"], "stage2_cv")
            if not stage2_summary.empty:
                update_manifest(
                    {
                        "stage2_completed_candidates": int(stage2_summary["candidate_id"].nunique()),
                        "stage2_total_candidates": len(stage2_candidates),
                        "stage2_best_cv_accuracy": float(stage2_summary["cv_mean_accuracy"].max()),
                    }
                )

    stage2_summary = read_dataframe(stage2_summary_path)
    if not stage2_summary.empty:
        display(stage2_summary.head(12))
        checkpoint_housekeeping("stage2_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


KNN_CLEANING_STAGE3 = dedent(
    """
    stage3_candidates_path = CHECKPOINT_DIR / "stage3_local_candidates.json"
    stage3_fold_path = CHECKPOINT_DIR / "stage3_local_cv_fold_results.csv"
    stage3_summary_path = CHECKPOINT_DIR / "stage3_local_cv_summary.csv"

    if stage2_summary.empty:
        raise RuntimeError("Stage 2 produced no results.")

    stage3_seed_rows = (
        stage2_summary.sort_values(["cv_mean_accuracy", "mean_removed_rows", "total_fit_seconds"], ascending=[False, True, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage3_seed_top_k"])
        .copy()
    )


    def get_or_create_stage3_candidates() -> list[dict]:
        if stage3_candidates_path.exists():
            return json.loads(stage3_candidates_path.read_text())

        seen = set()
        candidates = []
        next_index = 0
        for seed_rank, row in enumerate(stage3_seed_rows.to_dict(orient="records")):
            seed_params = json.loads(row["params_json"])
            pca_values = sorted({value for value in [seed_params["pca__n_components"] - 8, seed_params["pca__n_components"] - 4, seed_params["pca__n_components"], seed_params["pca__n_components"] + 4, seed_params["pca__n_components"] + 8] if 16 <= value <= X_train.shape[1]})
            neighbor_values = sorted({value for value in [seed_params["model__n_neighbors"] - 2, seed_params["model__n_neighbors"] - 1, seed_params["model__n_neighbors"], seed_params["model__n_neighbors"] + 1, seed_params["model__n_neighbors"] + 2, seed_params["model__n_neighbors"] + 4] if value >= 1})
            scale_values = sorted({seed_params["scale__method"], "standard", "robust"})
            metric_values = sorted({seed_params["model__metric"], "manhattan", "euclidean"})
            clean_values = sorted({seed_params["clean__method"], "none", "lof_0.03", "lof_0.05"})
            for clean_method in clean_values:
                for scale_method in scale_values:
                    for pca_components in pca_values:
                        for n_neighbors in neighbor_values:
                            for metric in metric_values:
                                params = normalize_candidate(
                                    {
                                        "clean__method": clean_method,
                                        "scale__method": scale_method,
                                        "pca__n_components": pca_components,
                                        "model__n_neighbors": n_neighbors,
                                        "model__metric": metric,
                                    }
                                )
                                signature = candidate_signature(params)
                                if signature in seen:
                                    continue
                                seen.add(signature)
                                candidates.append(candidate_record(f"stage3_{next_index:03d}", params, f"stage3_seed_{seed_rank}"))
                                next_index += 1

        write_json_atomic(stage3_candidates_path, candidates)
        return candidates


    stage3_candidates = get_or_create_stage3_candidates()
    stage3_cv = StratifiedKFold(n_splits=PROFILE["stage3_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage3_splits = list(stage3_cv.split(X_train, y_train))

    fold_df = read_dataframe(stage3_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()

    print("Stage 3 candidates:", len(stage3_candidates))
    print("Stage 3 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_3:
        for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(stage3_splits, desc="Stage 3 folds")):
            pending = [candidate for candidate in stage3_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            for candidate in tqdm(pending, desc=f"Stage 3 fold {fold_idx}", leave=False):
                accuracy, elapsed, removed_rows = evaluate_candidate(
                    X_train[fit_idx],
                    y_train[fit_idx],
                    X_train[eval_idx],
                    y_train[eval_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage3_local_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                    "removed_rows": removed_rows,
                }
                fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True) if not fold_df.empty else pd.DataFrame([row])
                save_dataframe_atomic(fold_df, stage3_fold_path)
                completed_pairs.add((candidate["candidate_id"], fold_idx))
                gc.collect()

            stage3_summary = refresh_summary_from_folds(stage3_fold_path, stage3_summary_path, stage3_candidates, PROFILE["stage3_cv"], "stage3_local_cv")
            if not stage3_summary.empty:
                update_manifest(
                    {
                        "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                        "stage3_total_candidates": len(stage3_candidates),
                        "stage3_best_cv_accuracy": float(stage3_summary["cv_mean_accuracy"].max()),
                    }
                )

    stage3_summary = read_dataframe(stage3_summary_path)
    if not stage3_summary.empty:
        display(stage3_summary.head(12))
        checkpoint_housekeeping("stage3_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


KNN_CLEANING_PLOTS = dedent(
    """
    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        stage1_plot = stage1_results.groupby("clean__method", as_index=False)["holdout_accuracy"].max().sort_values("holdout_accuracy", ascending=False)
        sns.barplot(data=stage1_plot, x="clean__method", y="holdout_accuracy")
        plt.xticks(rotation=35, ha="right")
        plt.title("Stage 1 best holdout accuracy by cleaning strategy")
        save_current_figure("stage1_cleaning_methods.png")
        display(stage1_plot)

    final_stage_df = stage3_summary if not stage3_summary.empty else stage2_summary if "stage2_summary" in globals() else stage1_results
    if final_stage_df is not None and not final_stage_df.empty:
        plt.figure(figsize=(12, 6))
        top_df = final_stage_df.head(15).copy()
        plot_col = "cv_mean_accuracy" if "cv_mean_accuracy" in top_df.columns else "holdout_accuracy"
        sns.barplot(data=top_df, x="candidate_id", y=plot_col, hue="clean__method")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top KNN cleaning candidates")
        save_current_figure("top_candidates.png")
    """
).strip()


KNN_CLEANING_FINAL = dedent(
    """
    final_summary_df = stage3_summary if not stage3_summary.empty else stage2_summary if not stage2_summary.empty else stage1_results
    if final_summary_df.empty:
        raise RuntimeError("No final candidate table is available.")

    best_row = final_summary_df.iloc[0].to_dict()
    best_candidate = candidate_record(best_row["candidate_id"], json.loads(best_row["params_json"]), "final_selection")

    final_bundle = fit_bundle(X_train, y_train, best_candidate)
    valid_predictions = predict_bundle(final_bundle, X_valid, proba=False)
    validation_accuracy = float(accuracy_score(y_valid, valid_predictions))

    cm = confusion_matrix(y_valid, valid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Validation confusion matrix - accuracy={validation_accuracy:.4f}")
    save_current_figure("validation_confusion_matrix.png")

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_1 = np.zeros(len(X_full), dtype=np.float32)
    test_prob_1 = np.zeros(len(X_test_full), dtype=np.float32)
    fold_records = []

    for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(oof_cv.split(X_full, y_full), desc="OOF for stacking")):
        fold_bundle = fit_bundle(X_full[fit_idx], y_full[fit_idx], best_candidate)
        oof_proba = predict_bundle(fold_bundle, X_full[eval_idx], proba=True)[:, 1]
        test_proba = predict_bundle(fold_bundle, X_test_full, proba=True)[:, 1]
        oof_prob_1[eval_idx] = oof_proba.astype(np.float32)
        test_prob_1 += test_proba.astype(np.float32) / oof_cv.n_splits
        fold_records.append(
            {
                "fold_idx": fold_idx,
                "fold_accuracy": float(accuracy_score(y_full[eval_idx], (oof_proba >= 0.5).astype(int))),
            }
        )

    final_fit_bundle = fit_bundle(X_full, y_full, best_candidate)
    final_test_pred = predict_bundle(final_fit_bundle, X_test_full, proba=False).astype(int)

    oof_df = pd.DataFrame(
        {
            "id": train_ids,
            "y_true": y_full.astype(int),
            "prob_1": oof_prob_1,
            "pred": (oof_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    test_prob_df = pd.DataFrame(
        {
            "id": test_ids,
            "prob_1": test_prob_1,
            "pred": (test_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    submission_df = sample_df.copy()
    submission_df["class"] = final_test_pred.astype(int)

    oof_path = PERSIST_ROOT / "oof_probabilities.csv"
    test_prob_path = PERSIST_ROOT / "test_probabilities.csv"
    fold_summary_path = PERSIST_ROOT / "oof_fold_summary.csv"
    submission_path = SUBMISSION_DIR / "challenge_08_knn_cleaning_colab_ultra_submission.csv"
    summary_path = PERSIST_ROOT / "summary.json"

    save_dataframe_atomic(oof_df, oof_path)
    save_dataframe_atomic(test_prob_df, test_prob_path)
    save_dataframe_atomic(pd.DataFrame(fold_records), fold_summary_path)
    submission_df.to_csv(submission_path, index=False)

    summary_payload = {
        "model_name": "KNN with instance cleaning",
        "model_key": "knn_cleaning",
        "notebook_slug": NOTEBOOK_SLUG,
        "strategy": "colab_ultra_cleaning_search",
        "search_profile": SEARCH_PROFILE,
        "best_stage": str(best_row.get("stage", "unknown")),
        "best_params": best_candidate["params"],
        "validation_accuracy": validation_accuracy,
        "validation_confusion_matrix": cm.tolist(),
        "oof_accuracy": float(accuracy_score(y_full, (oof_prob_1 >= 0.5).astype(int))),
        "oof_path": str(oof_path),
        "test_probability_path": str(test_prob_path),
        "submission_path": str(submission_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "persist_root": str(PERSIST_ROOT),
    }
    write_json_atomic(summary_path, summary_payload)
    checkpoint_housekeeping("final_model_complete", refresh_bundle=True, include_data_in_bundle=False)

    print("Best params:", json.dumps(best_candidate["params"], indent=2))
    print("Validation accuracy:", validation_accuracy)
    print("OOF accuracy:", summary_payload["oof_accuracy"])
    print("Submission path:", submission_path)
    """
).strip()


KNN_CLEANING_OUTRO = dedent(
    """
    ## Notes

    Esta notebook deja listos tres artefactos clave para una fase posterior de stacking:

    - `summary.json`
    - `oof_probabilities.csv`
    - `test_probabilities.csv`

    Si el score publico mejora, esta variante debe pasar a ser base learner del stacking final.
    """
).strip()


SVM_PREP_INTRO = dedent(
    """
    # Challenge 09 Colab Ultra: SVM with preprocessing search

    Esta variante asume que el espacio util de `SVM` ya esta bastante claro.

    Por eso el objetivo ya no es otro grid search enorme de `C` y `gamma`, sino evaluar si el salto viene de:

    - un mejor escalado
    - una limpieza suave de instancias
    - una mejor transformacion de la distribucion de las variables

    ## Estrategia

    1. `Stage 1`: screening amplio de pipelines de preprocesamiento + SVM.
    2. `Stage 2`: shortlist con `3-fold CV`.
    3. `Stage 3`: refinamiento local solo alrededor de la mejor region.
    4. Modelo final, submission y artefactos para stacking.
    """
).strip()


SVM_PREP_IMPORTS = common_imports(
    "challenge_09_svm_preprocessing_colab_ultra",
    dedent(
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
        from sklearn.svm import SVC
        """
    ).strip(),
)


SVM_PREP_RUNTIME = dedent(
    """
    CPU_COUNT = os.cpu_count() or 2
    RAM_GB = None if psutil is None else round(psutil.virtual_memory().total / (1024 ** 3), 2)
    SVM_CACHE_MB = 1024 if (RAM_GB is not None and RAM_GB >= 20) else 512

    SEARCH_PRESETS = {
        "balanced": {
            "stage1_n_iter": 64,
            "stage1_pool_fraction": 0.70,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 14,
            "stage2_cv": 3,
            "stage3_seed_top_k": 3,
            "stage3_cv": 5,
        },
        "aggressive": {
            "stage1_n_iter": 96,
            "stage1_pool_fraction": 0.82,
            "stage1_eval_size": 0.25,
            "stage2_top_k": 18,
            "stage2_cv": 3,
            "stage3_seed_top_k": 4,
            "stage3_cv": 5,
        },
    }

    SEARCH_PROFILE = "aggressive"
    PROFILE = SEARCH_PRESETS[SEARCH_PROFILE]

    RUN_STAGE_1 = True
    RUN_STAGE_2 = True
    RUN_STAGE_3 = True
    TRAIN_FINAL_MODEL = True

    print(
        {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "cpu_count": CPU_COUNT,
            "ram_gb": RAM_GB,
            "svm_cache_mb": SVM_CACHE_MB,
            "search_profile": SEARCH_PROFILE,
        }
    )
    """
).strip()


SVM_PREP_HELPERS = dedent(
    """
    CLEANING_METHODS = ["none", "lof_0.03", "lof_0.05"]
    SCALER_METHODS = ["standard", "robust", "quantile_normal", "power_yeo"]


    def normalize_candidate(params: dict) -> dict:
        normalized = {key: normalize_value(value) for key, value in params.items()}
        normalized["clean__method"] = str(normalized["clean__method"])
        normalized["scale__method"] = str(normalized["scale__method"])
        normalized["pca__n_components"] = int(normalized["pca__n_components"])
        normalized["model__C"] = float(normalized["model__C"])
        normalized["model__gamma"] = float(normalized["model__gamma"])
        return normalized


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
            "clean__method": params["clean__method"],
            "scale__method": params["scale__method"],
            "pca__n_components": params["pca__n_components"],
            "model__C": params["model__C"],
            "model__gamma": params["model__gamma"],
            "params_json": json.dumps(params, sort_keys=True),
        }


    def apply_lof_cleaning(X_input: np.ndarray, y_input: np.ndarray, contamination: float) -> tuple[np.ndarray, np.ndarray, int]:
        scaled = StandardScaler().fit_transform(X_input)
        keep_mask = np.ones(len(X_input), dtype=bool)
        for cls in np.unique(y_input):
            idx = np.where(y_input == cls)[0]
            if len(idx) < 12:
                continue
            n_neighbors = min(25, len(idx) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            pred = lof.fit_predict(scaled[idx])
            keep_mask[idx] = pred == 1
        return X_input[keep_mask], y_input[keep_mask], int((~keep_mask).sum())


    def apply_cleaning(X_input: np.ndarray, y_input: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if method == "none":
            return X_input, y_input, {"removed_rows": 0}
        contamination = float(method.split("_")[1])
        X_clean, y_clean, removed = apply_lof_cleaning(X_input, y_input, contamination)
        return X_clean, y_clean, {"removed_rows": removed}


    def choose_scaler(name: str, n_samples: int):
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        if name == "quantile_normal":
            return QuantileTransformer(n_quantiles=min(1000, max(10, n_samples)), output_distribution="normal", random_state=RANDOM_STATE)
        if name == "power_yeo":
            return PowerTransformer(method="yeo-johnson")
        raise ValueError(f"Unknown scaler: {name}")


    def build_svm(candidate: dict, probability: bool = False) -> SVC:
        params = candidate["params"]
        return SVC(
            kernel="rbf",
            C=float(params["model__C"]),
            gamma=float(params["model__gamma"]),
            cache_size=SVM_CACHE_MB,
            shrinking=True,
            probability=probability,
        )


    def fit_bundle(X_fit: np.ndarray, y_fit: np.ndarray, candidate: dict, probability: bool = False) -> dict:
        params = candidate["params"]
        X_clean, y_clean, cleaning_info = apply_cleaning(X_fit, y_fit, params["clean__method"])
        scaler = choose_scaler(params["scale__method"], len(X_clean))
        X_clean_scaled = scaler.fit_transform(X_clean)
        n_components = min(int(params["pca__n_components"]), X_clean_scaled.shape[0], X_clean_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_clean_proj = pca.fit_transform(X_clean_scaled).astype(np.float32)
        model = build_svm(candidate, probability=probability)
        model.fit(X_clean_proj, y_clean)
        return {
            "candidate": candidate,
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "cleaning_info": cleaning_info,
        }


    def predict_bundle(bundle: dict, X_input: np.ndarray, proba: bool = False) -> np.ndarray:
        X_scaled = bundle["scaler"].transform(X_input)
        X_proj = bundle["pca"].transform(X_scaled).astype(np.float32)
        if proba:
            return bundle["model"].predict_proba(X_proj)
        return bundle["model"].predict(X_proj)


    def evaluate_candidate(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, candidate: dict) -> tuple[float, float, int]:
        start = time.time()
        bundle = fit_bundle(X_fit, y_fit, candidate, probability=False)
        predictions = predict_bundle(bundle, X_eval, proba=False)
        accuracy = accuracy_score(y_eval, predictions)
        elapsed = round(time.time() - start, 3)
        removed_rows = int(bundle["cleaning_info"].get("removed_rows", 0))
        return float(accuracy), elapsed, removed_rows


    def refresh_summary_from_folds(fold_path: Path, summary_path: Path, candidates: list[dict], n_splits: int, stage_name: str) -> pd.DataFrame:
        fold_df = read_dataframe(fold_path)
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for candidate in candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if candidate_fold_df["fold_idx"].nunique() < n_splits:
                continue
            row = {
                **candidate_to_row(candidate),
                "stage": stage_name,
                "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                "mean_removed_rows": float(candidate_fold_df["removed_rows"].mean()),
                "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "mean_removed_rows", "total_fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, summary_path)
        return summary_df
    """
).strip()


SVM_PREP_STAGE1 = dedent(
    """
    STAGE1_SPACE = {
        "clean__method": CLEANING_METHODS,
        "scale__method": SCALER_METHODS,
        "pca__n_components": [96, 104, 112, 120, 128, 136, 144],
        "model__C": [2.0, 3.0, 4.0, 6.0, 8.0],
        "model__gamma": [0.005, 0.01, 0.02],
    }

    stage1_candidates_path = CHECKPOINT_DIR / "stage1_candidates.json"
    stage1_results_path = CHECKPOINT_DIR / "stage1_holdout_results.csv"


    def get_or_create_stage1_candidates() -> list[dict]:
        if stage1_candidates_path.exists():
            return json.loads(stage1_candidates_path.read_text())

        sampler = ParameterSampler(STAGE1_SPACE, n_iter=PROFILE["stage1_n_iter"], random_state=RANDOM_STATE)
        candidates = []
        seen = set()
        for sampled_params in sampler:
            normalized = normalize_candidate(sampled_params)
            signature = candidate_signature(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate_record(f"stage1_{len(candidates):03d}", normalized, "stage1_random_holdout"))
        write_json_atomic(stage1_candidates_path, candidates)
        return candidates


    X_stage1_pool, _, y_stage1_pool, _ = train_test_split(
        X_train,
        y_train,
        train_size=PROFILE["stage1_pool_fraction"],
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_stage1_fit, X_stage1_eval, y_stage1_fit, y_stage1_eval = train_test_split(
        X_stage1_pool,
        y_stage1_pool,
        test_size=PROFILE["stage1_eval_size"],
        stratify=y_stage1_pool,
        random_state=RANDOM_STATE,
    )

    stage1_candidates = get_or_create_stage1_candidates()
    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    print("Stage 1 candidates:", len(stage1_candidates))
    print("Stage 1 completed:", len(completed_stage1))

    if RUN_STAGE_1:
        for candidate in tqdm([c for c in stage1_candidates if c["candidate_id"] not in completed_stage1], desc="Stage 1 screening"):
            accuracy, elapsed, removed_rows = evaluate_candidate(X_stage1_fit, y_stage1_fit, X_stage1_eval, y_stage1_eval, candidate)
            row = {
                **candidate_to_row(candidate),
                "stage": "stage1_holdout",
                "holdout_accuracy": accuracy,
                "fit_seconds": elapsed,
                "removed_rows": removed_rows,
            }
            stage1_results = pd.concat([stage1_results, pd.DataFrame([row])], ignore_index=True) if not stage1_results.empty else pd.DataFrame([row])
            stage1_results = stage1_results.sort_values(["holdout_accuracy", "removed_rows", "fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(stage1_results, stage1_results_path)
            update_manifest(
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_holdout_accuracy": float(stage1_results["holdout_accuracy"].max()),
                }
            )
            gc.collect()

    stage1_results = read_dataframe(stage1_results_path)
    if not stage1_results.empty:
        display(stage1_results.head(12))
        checkpoint_housekeeping("stage1_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SVM_PREP_STAGE2 = dedent(
    """
    stage2_fold_path = CHECKPOINT_DIR / "stage2_cv_fold_results.csv"
    stage2_summary_path = CHECKPOINT_DIR / "stage2_cv_summary.csv"

    if stage1_results.empty:
        raise RuntimeError("Stage 1 produced no results.")

    stage2_shortlist = (
        stage1_results.sort_values(["holdout_accuracy", "removed_rows", "fit_seconds"], ascending=[False, True, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage2_top_k"])
        .copy()
    )
    stage2_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage2_shortlist_from_stage1")
        for row in stage2_shortlist.to_dict(orient="records")
    ]

    stage2_cv = StratifiedKFold(n_splits=PROFILE["stage2_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage2_splits = list(stage2_cv.split(X_train, y_train))

    fold_df = read_dataframe(stage2_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()

    print("Stage 2 candidates:", len(stage2_candidates))
    print("Stage 2 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_2:
        for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(stage2_splits, desc="Stage 2 folds")):
            pending = [candidate for candidate in stage2_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            for candidate in tqdm(pending, desc=f"Stage 2 fold {fold_idx}", leave=False):
                accuracy, elapsed, removed_rows = evaluate_candidate(
                    X_train[fit_idx],
                    y_train[fit_idx],
                    X_train[eval_idx],
                    y_train[eval_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                    "removed_rows": removed_rows,
                }
                fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True) if not fold_df.empty else pd.DataFrame([row])
                save_dataframe_atomic(fold_df, stage2_fold_path)
                completed_pairs.add((candidate["candidate_id"], fold_idx))
                gc.collect()

            stage2_summary = refresh_summary_from_folds(stage2_fold_path, stage2_summary_path, stage2_candidates, PROFILE["stage2_cv"], "stage2_cv")
            if not stage2_summary.empty:
                update_manifest(
                    {
                        "stage2_completed_candidates": int(stage2_summary["candidate_id"].nunique()),
                        "stage2_total_candidates": len(stage2_candidates),
                        "stage2_best_cv_accuracy": float(stage2_summary["cv_mean_accuracy"].max()),
                    }
                )

    stage2_summary = read_dataframe(stage2_summary_path)
    if not stage2_summary.empty:
        display(stage2_summary.head(12))
        checkpoint_housekeeping("stage2_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SVM_PREP_STAGE3 = dedent(
    """
    stage3_candidates_path = CHECKPOINT_DIR / "stage3_local_candidates.json"
    stage3_fold_path = CHECKPOINT_DIR / "stage3_local_cv_fold_results.csv"
    stage3_summary_path = CHECKPOINT_DIR / "stage3_local_cv_summary.csv"

    if stage2_summary.empty:
        raise RuntimeError("Stage 2 produced no results.")

    stage3_seed_rows = (
        stage2_summary.sort_values(["cv_mean_accuracy", "mean_removed_rows", "total_fit_seconds"], ascending=[False, True, True])
        .drop_duplicates("signature")
        .head(PROFILE["stage3_seed_top_k"])
        .copy()
    )


    def get_or_create_stage3_candidates() -> list[dict]:
        if stage3_candidates_path.exists():
            return json.loads(stage3_candidates_path.read_text())

        seen = set()
        candidates = []
        next_index = 0
        for seed_rank, row in enumerate(stage3_seed_rows.to_dict(orient="records")):
            seed = json.loads(row["params_json"])
            pca_values = sorted({value for value in [seed["pca__n_components"] - 16, seed["pca__n_components"] - 8, seed["pca__n_components"], seed["pca__n_components"] + 8, seed["pca__n_components"] + 16] if 48 <= value <= X_train.shape[1]})
            c_values = sorted({round(max(0.5, seed["model__C"] * ratio), 4) for ratio in [0.75, 1.0, 1.5, 2.0]})
            gamma_values = sorted({round(max(0.001, seed["model__gamma"] * ratio), 5) for ratio in [0.5, 1.0, 1.5, 2.0] if max(0.001, seed["model__gamma"] * ratio) <= 0.05})
            clean_values = sorted({seed["clean__method"], "none", "lof_0.03", "lof_0.05"})
            scale_values = sorted({seed["scale__method"], "standard", "robust", "quantile_normal", "power_yeo"})
            for clean_method in clean_values:
                for scale_method in scale_values:
                    for pca_components in pca_values:
                        for c_value in c_values:
                            for gamma_value in gamma_values:
                                params = normalize_candidate(
                                    {
                                        "clean__method": clean_method,
                                        "scale__method": scale_method,
                                        "pca__n_components": pca_components,
                                        "model__C": c_value,
                                        "model__gamma": gamma_value,
                                    }
                                )
                                signature = candidate_signature(params)
                                if signature in seen:
                                    continue
                                seen.add(signature)
                                candidates.append(candidate_record(f"stage3_{next_index:03d}", params, f"stage3_seed_{seed_rank}"))
                                next_index += 1
        write_json_atomic(stage3_candidates_path, candidates)
        return candidates


    stage3_candidates = get_or_create_stage3_candidates()
    stage3_cv = StratifiedKFold(n_splits=PROFILE["stage3_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage3_splits = list(stage3_cv.split(X_train, y_train))

    fold_df = read_dataframe(stage3_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()

    print("Stage 3 candidates:", len(stage3_candidates))
    print("Stage 3 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_3:
        for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(stage3_splits, desc="Stage 3 folds")):
            pending = [candidate for candidate in stage3_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            for candidate in tqdm(pending, desc=f"Stage 3 fold {fold_idx}", leave=False):
                accuracy, elapsed, removed_rows = evaluate_candidate(
                    X_train[fit_idx],
                    y_train[fit_idx],
                    X_train[eval_idx],
                    y_train[eval_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage3_local_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                    "removed_rows": removed_rows,
                }
                fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True) if not fold_df.empty else pd.DataFrame([row])
                save_dataframe_atomic(fold_df, stage3_fold_path)
                completed_pairs.add((candidate["candidate_id"], fold_idx))
                gc.collect()

            stage3_summary = refresh_summary_from_folds(stage3_fold_path, stage3_summary_path, stage3_candidates, PROFILE["stage3_cv"], "stage3_local_cv")
            if not stage3_summary.empty:
                update_manifest(
                    {
                        "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                        "stage3_total_candidates": len(stage3_candidates),
                        "stage3_best_cv_accuracy": float(stage3_summary["cv_mean_accuracy"].max()),
                    }
                )

    stage3_summary = read_dataframe(stage3_summary_path)
    if not stage3_summary.empty:
        display(stage3_summary.head(12))
        checkpoint_housekeeping("stage3_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SVM_PREP_PLOTS = dedent(
    """
    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        stage1_plot = stage1_results.groupby("scale__method", as_index=False)["holdout_accuracy"].max().sort_values("holdout_accuracy", ascending=False)
        sns.barplot(data=stage1_plot, x="scale__method", y="holdout_accuracy")
        plt.xticks(rotation=25, ha="right")
        plt.title("Stage 1 best holdout accuracy by preprocessing scaler")
        save_current_figure("stage1_scaler_methods.png")
        display(stage1_plot)

    final_stage_df = stage3_summary if not stage3_summary.empty else stage2_summary if "stage2_summary" in globals() else stage1_results
    if final_stage_df is not None and not final_stage_df.empty:
        plt.figure(figsize=(10, 7))
        plot_df = final_stage_df.head(20).copy()
        sns.scatterplot(data=plot_df, x="pca__n_components", y="cv_mean_accuracy" if "cv_mean_accuracy" in plot_df.columns else "holdout_accuracy", hue="scale__method", style="clean__method", s=120)
        plt.title("Top SVM preprocessing candidates")
        save_current_figure("top_candidates.png")
    """
).strip()


SVM_PREP_FINAL = dedent(
    """
    final_summary_df = stage3_summary if not stage3_summary.empty else stage2_summary if not stage2_summary.empty else stage1_results
    if final_summary_df.empty:
        raise RuntimeError("No final candidate table is available.")

    best_row = final_summary_df.iloc[0].to_dict()
    best_candidate = candidate_record(best_row["candidate_id"], json.loads(best_row["params_json"]), "final_selection")

    final_bundle = fit_bundle(X_train, y_train, best_candidate, probability=False)
    valid_predictions = predict_bundle(final_bundle, X_valid, proba=False)
    validation_accuracy = float(accuracy_score(y_valid, valid_predictions))

    cm = confusion_matrix(y_valid, valid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Validation confusion matrix - accuracy={validation_accuracy:.4f}")
    save_current_figure("validation_confusion_matrix.png")

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_1 = np.zeros(len(X_full), dtype=np.float32)
    test_prob_1 = np.zeros(len(X_test_full), dtype=np.float32)
    fold_records = []

    for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(oof_cv.split(X_full, y_full), desc="OOF for stacking")):
        fold_bundle = fit_bundle(X_full[fit_idx], y_full[fit_idx], best_candidate, probability=True)
        oof_proba = predict_bundle(fold_bundle, X_full[eval_idx], proba=True)[:, 1]
        test_proba = predict_bundle(fold_bundle, X_test_full, proba=True)[:, 1]
        oof_prob_1[eval_idx] = oof_proba.astype(np.float32)
        test_prob_1 += test_proba.astype(np.float32) / oof_cv.n_splits
        fold_records.append(
            {
                "fold_idx": fold_idx,
                "fold_accuracy": float(accuracy_score(y_full[eval_idx], (oof_proba >= 0.5).astype(int))),
            }
        )
        gc.collect()

    final_fit_bundle = fit_bundle(X_full, y_full, best_candidate, probability=False)
    final_test_pred = predict_bundle(final_fit_bundle, X_test_full, proba=False).astype(int)

    oof_df = pd.DataFrame(
        {
            "id": train_ids,
            "y_true": y_full.astype(int),
            "prob_1": oof_prob_1,
            "pred": (oof_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    test_prob_df = pd.DataFrame(
        {
            "id": test_ids,
            "prob_1": test_prob_1,
            "pred": (test_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    submission_df = sample_df.copy()
    submission_df["class"] = final_test_pred.astype(int)

    oof_path = PERSIST_ROOT / "oof_probabilities.csv"
    test_prob_path = PERSIST_ROOT / "test_probabilities.csv"
    fold_summary_path = PERSIST_ROOT / "oof_fold_summary.csv"
    submission_path = SUBMISSION_DIR / "challenge_09_svm_preprocessing_colab_ultra_submission.csv"
    summary_path = PERSIST_ROOT / "summary.json"

    save_dataframe_atomic(oof_df, oof_path)
    save_dataframe_atomic(test_prob_df, test_prob_path)
    save_dataframe_atomic(pd.DataFrame(fold_records), fold_summary_path)
    submission_df.to_csv(submission_path, index=False)

    summary_payload = {
        "model_name": "SVM with preprocessing search",
        "model_key": "svm_preprocessing",
        "notebook_slug": NOTEBOOK_SLUG,
        "strategy": "colab_ultra_preprocessing_search",
        "search_profile": SEARCH_PROFILE,
        "best_stage": str(best_row.get("stage", "unknown")),
        "best_params": best_candidate["params"],
        "validation_accuracy": validation_accuracy,
        "validation_confusion_matrix": cm.tolist(),
        "oof_accuracy": float(accuracy_score(y_full, (oof_prob_1 >= 0.5).astype(int))),
        "oof_path": str(oof_path),
        "test_probability_path": str(test_prob_path),
        "submission_path": str(submission_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "persist_root": str(PERSIST_ROOT),
    }
    write_json_atomic(summary_path, summary_payload)
    checkpoint_housekeeping("final_model_complete", refresh_bundle=True, include_data_in_bundle=False)

    print("Best params:", json.dumps(best_candidate["params"], indent=2))
    print("Validation accuracy:", validation_accuracy)
    print("OOF accuracy:", summary_payload["oof_accuracy"])
    print("Submission path:", submission_path)
    """
).strip()


SVM_PREP_OUTRO = dedent(
    """
    ## Notes

    Esta notebook esta diseñada para responder una pregunta especifica:

    > En SVM, ya no conviene seguir abriendo el espacio de `C/gamma`; conviene probar mejor preprocesamiento.

    Tambien deja listos los artefactos para stacking:

    - `oof_probabilities.csv`
    - `test_probabilities.csv`
    """
).strip()


SIGNAL_FEATURES_INTRO = dedent(
    """
    # Challenge 10 Colab Ultra: signal feature engineering

    Esta variante explota la hipotesis mas importante que aun no habiamos incorporado al pipeline:

    - los predictores `V1 ... V200` muy probablemente representan una señal ordenada
    - si eso es cierto, el mejor salto puede venir de `feature engineering` de señal y no de otro algoritmo

    ## Estrategia

    1. Construir bancos de features en dominio temporal y frecuencial.
    2. Evaluar candidatos `KNN` y `SVM` sobre esas features.
    3. Probar tambien una version hibrida que concatena features ingenierizadas con `raw PCA`.
    4. Exportar la mejor variante para submission y stacking.
    """
).strip()


SIGNAL_FEATURES_IMPORTS = common_imports(
    "challenge_10_signal_features_colab_ultra",
    dedent(
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import RobustScaler, StandardScaler
        from sklearn.svm import SVC
        """
    ).strip(),
)


SIGNAL_FEATURES_RUNTIME = dedent(
    """
    CPU_COUNT = os.cpu_count() or 2
    RAM_GB = None if psutil is None else round(psutil.virtual_memory().total / (1024 ** 3), 2)
    SVM_CACHE_MB = 1024 if (RAM_GB is not None and RAM_GB >= 20) else 512
    KNN_N_JOBS = max(1, CPU_COUNT - 1)

    SEARCH_PRESETS = {
        "balanced": {"stage1_max_candidates": 54, "stage2_top_k": 12, "stage2_cv": 3},
        "aggressive": {"stage1_max_candidates": 84, "stage2_top_k": 18, "stage2_cv": 5},
    }

    SEARCH_PROFILE = "aggressive"
    PROFILE = SEARCH_PRESETS[SEARCH_PROFILE]

    RUN_STAGE_1 = True
    RUN_STAGE_2 = True
    TRAIN_FINAL_MODEL = True

    print(
        {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "cpu_count": CPU_COUNT,
            "ram_gb": RAM_GB,
            "search_profile": SEARCH_PROFILE,
        }
    )
    """
).strip()


SIGNAL_FEATURES_HELPERS = dedent(
    """
    FEATURE_SET_NAMES = ["basic", "fft", "all"]


    def normalize_candidate(params: dict) -> dict:
        normalized = {key: normalize_value(value) for key, value in params.items()}
        normalized["feature__set"] = str(normalized["feature__set"])
        normalized["feature__raw_pca"] = int(normalized["feature__raw_pca"])
        normalized["scale__method"] = str(normalized["scale__method"])
        normalized["model__family"] = str(normalized["model__family"])
        if normalized["model__family"] == "knn":
            normalized["model__n_neighbors"] = int(normalized["model__n_neighbors"])
            normalized["model__metric"] = str(normalized["model__metric"])
            normalized["model__C"] = None
            normalized["model__gamma"] = None
        else:
            normalized["model__n_neighbors"] = None
            normalized["model__metric"] = None
            normalized["model__C"] = float(normalized["model__C"])
            normalized["model__gamma"] = float(normalized["model__gamma"])
        return normalized


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
            "feature__set": params["feature__set"],
            "feature__raw_pca": params["feature__raw_pca"],
            "scale__method": params["scale__method"],
            "model__family": params["model__family"],
            "model__n_neighbors": params["model__n_neighbors"],
            "model__metric": params["model__metric"],
            "model__C": params["model__C"],
            "model__gamma": params["model__gamma"],
            "params_json": json.dumps(params, sort_keys=True),
        }


    def choose_scaler(name: str):
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler: {name}")


    def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=np.abs(den) > 1e-12)


    def build_feature_bank(X_input: np.ndarray) -> dict[str, np.ndarray]:
        X = X_input.astype(np.float64)
        eps = 1e-12
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        minimum = X.min(axis=1)
        maximum = X.max(axis=1)
        median = np.median(X, axis=1)
        q25 = np.quantile(X, 0.25, axis=1)
        q75 = np.quantile(X, 0.75, axis=1)
        iqr = q75 - q25
        rms = np.sqrt(np.mean(X ** 2, axis=1))
        abs_mean = np.mean(np.abs(X), axis=1)
        max_abs = np.max(np.abs(X), axis=1)
        peak_to_peak = maximum - minimum
        centered = X - mean[:, None]
        skew = _safe_divide(np.mean(centered ** 3, axis=1), (std ** 3) + eps)
        kurt = _safe_divide(np.mean(centered ** 4, axis=1), (std ** 4) + eps)
        zero_cross = np.mean((X[:, 1:] * X[:, :-1]) < 0, axis=1)
        energy = np.mean(X ** 2, axis=1)
        diffs = np.diff(X, axis=1)
        diff_energy = np.mean(diffs ** 2, axis=1)
        corr1 = _safe_divide(np.sum(centered[:, 1:] * centered[:, :-1], axis=1), np.sum(centered[:, :-1] ** 2, axis=1) + eps)
        corr2 = _safe_divide(np.sum(centered[:, 2:] * centered[:, :-2], axis=1), np.sum(centered[:, :-2] ** 2, axis=1) + eps)
        corr4 = _safe_divide(np.sum(centered[:, 4:] * centered[:, :-4], axis=1), np.sum(centered[:, :-4] ** 2, axis=1) + eps)

        basic = np.column_stack(
            [
                mean, std, minimum, maximum, median, q25, q75, iqr, rms,
                abs_mean, max_abs, peak_to_peak, skew, kurt, zero_cross,
                energy, diff_energy, corr1, corr2, corr4,
            ]
        )

        fft_mag = np.abs(np.fft.rfft(X, axis=1))
        fft_power = fft_mag ** 2
        freq_idx = np.arange(fft_mag.shape[1], dtype=np.float64)
        total_mag = fft_mag.sum(axis=1) + eps
        spectral_centroid = np.sum(fft_mag * freq_idx[None, :], axis=1) / total_mag
        dominant_idx = np.argmax(fft_mag[:, 1:], axis=1) + 1
        dominant_amp = np.max(fft_mag[:, 1:], axis=1)
        p = fft_power / (fft_power.sum(axis=1, keepdims=True) + eps)
        spectral_entropy = -np.sum(p * np.log(p + eps), axis=1)
        split_points = np.linspace(0, fft_mag.shape[1], 5, dtype=int)
        band_energy = []
        for left, right in zip(split_points[:-1], split_points[1:]):
            band_energy.append(fft_power[:, left:right].sum(axis=1))
        fft_features = np.column_stack([spectral_centroid, dominant_idx, dominant_amp, spectral_entropy, *band_energy])

        segments = np.array_split(X, 4, axis=1)
        seg_stats = []
        for seg in segments:
            seg_stats.extend(
                [
                    seg.mean(axis=1),
                    seg.std(axis=1),
                    np.sqrt(np.mean(seg ** 2, axis=1)),
                    np.max(np.abs(seg), axis=1),
                ]
            )
        segment_features = np.column_stack(seg_stats)

        return {
            "basic": basic.astype(np.float32),
            "fft": np.hstack([basic, fft_features]).astype(np.float32),
            "all": np.hstack([basic, fft_features, segment_features]).astype(np.float32),
        }


    FEATURE_BANK_CACHE = {}


    def get_feature_bank(cache_key: str, X_input: np.ndarray) -> dict[str, np.ndarray]:
        if cache_key not in FEATURE_BANK_CACHE:
            FEATURE_BANK_CACHE[cache_key] = build_feature_bank(X_input)
        return FEATURE_BANK_CACHE[cache_key]


    def build_matrix(X_fit: np.ndarray, X_eval: np.ndarray, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
        params = candidate["params"]
        fit_bank = build_feature_bank(X_fit)
        eval_bank = build_feature_bank(X_eval)
        X_fit_base = fit_bank[params["feature__set"]]
        X_eval_base = eval_bank[params["feature__set"]]
        if params["feature__raw_pca"] > 0:
            raw_scaler = StandardScaler()
            X_fit_raw_scaled = raw_scaler.fit_transform(X_fit)
            X_eval_raw_scaled = raw_scaler.transform(X_eval)
            raw_pca = PCA(n_components=min(params["feature__raw_pca"], X_fit_raw_scaled.shape[0], X_fit_raw_scaled.shape[1]), random_state=RANDOM_STATE)
            X_fit_raw = raw_pca.fit_transform(X_fit_raw_scaled).astype(np.float32)
            X_eval_raw = raw_pca.transform(X_eval_raw_scaled).astype(np.float32)
            X_fit_base = np.hstack([X_fit_base, X_fit_raw]).astype(np.float32)
            X_eval_base = np.hstack([X_eval_base, X_eval_raw]).astype(np.float32)
        scaler = choose_scaler(params["scale__method"])
        X_fit_scaled = scaler.fit_transform(X_fit_base).astype(np.float32)
        X_eval_scaled = scaler.transform(X_eval_base).astype(np.float32)
        return X_fit_scaled, X_eval_scaled


    def build_model(candidate: dict, probability: bool = False):
        params = candidate["params"]
        if params["model__family"] == "knn":
            return KNeighborsClassifier(
                n_neighbors=int(params["model__n_neighbors"]),
                metric=params["model__metric"],
                weights="distance",
                algorithm="brute",
                n_jobs=KNN_N_JOBS,
            )
        return SVC(
            kernel="rbf",
            C=float(params["model__C"]),
            gamma=float(params["model__gamma"]),
            cache_size=SVM_CACHE_MB,
            probability=probability,
        )


    def fit_bundle(X_fit: np.ndarray, y_fit: np.ndarray, candidate: dict, probability: bool = False) -> dict:
        X_fit_matrix, _ = build_matrix(X_fit, X_fit[: min(2, len(X_fit))], candidate)
        model = build_model(candidate, probability=probability)
        model.fit(X_fit_matrix, y_fit)
        return {"candidate": candidate, "model": model}


    def predict_with_bundle(bundle: dict, X_fit_reference: np.ndarray, X_input: np.ndarray, proba: bool = False) -> np.ndarray:
        X_fit_matrix, X_eval_matrix = build_matrix(X_fit_reference, X_input, bundle["candidate"])
        if X_fit_matrix.shape[0] != len(X_fit_reference):
            raise RuntimeError("Unexpected matrix shape during prediction.")
        if proba:
            return bundle["model"].predict_proba(X_eval_matrix)
        return bundle["model"].predict(X_eval_matrix)


    def evaluate_candidate(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, candidate: dict) -> tuple[float, float]:
        start = time.time()
        X_fit_matrix, X_eval_matrix = build_matrix(X_fit, X_eval, candidate)
        model = build_model(candidate, probability=False)
        model.fit(X_fit_matrix, y_fit)
        predictions = model.predict(X_eval_matrix)
        accuracy = accuracy_score(y_eval, predictions)
        elapsed = round(time.time() - start, 3)
        return float(accuracy), elapsed


    def refresh_summary_from_folds(fold_path: Path, summary_path: Path, candidates: list[dict], n_splits: int, stage_name: str) -> pd.DataFrame:
        fold_df = read_dataframe(fold_path)
        if fold_df.empty:
            return pd.DataFrame()
        rows = []
        for candidate in candidates:
            candidate_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if candidate_df["fold_idx"].nunique() < n_splits:
                continue
            rows.append(
                {
                    **candidate_to_row(candidate),
                    "stage": stage_name,
                    "cv_mean_accuracy": float(candidate_df["fold_accuracy"].mean()),
                    "cv_std_accuracy": float(candidate_df["fold_accuracy"].std(ddof=0)),
                    "total_fit_seconds": float(candidate_df["fit_seconds"].sum()),
                }
            )
        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, summary_path)
        return summary_df
    """
).strip()


SIGNAL_FEATURES_STAGE1 = dedent(
    """
    candidate_list = []
    next_index = 0
    for feature_set in FEATURE_SET_NAMES:
        for raw_pca in [0, 16, 32]:
            for scale_method in ["standard", "robust"]:
                for k_value, metric_value in [(3, "manhattan"), (5, "manhattan"), (7, "euclidean")]:
                    candidate_list.append(
                        candidate_record(
                            f"stage1_{next_index:03d}",
                            {
                                "feature__set": feature_set,
                                "feature__raw_pca": raw_pca,
                                "scale__method": scale_method,
                                "model__family": "knn",
                                "model__n_neighbors": k_value,
                                "model__metric": metric_value,
                                "model__C": None,
                                "model__gamma": None,
                            },
                            "stage1_manual_bank",
                        )
                    )
                    next_index += 1
                for c_value, gamma_value in [(2.0, 0.005), (3.0, 0.01), (6.0, 0.01)]:
                    candidate_list.append(
                        candidate_record(
                            f"stage1_{next_index:03d}",
                            {
                                "feature__set": feature_set,
                                "feature__raw_pca": raw_pca,
                                "scale__method": scale_method,
                                "model__family": "svm",
                                "model__n_neighbors": None,
                                "model__metric": None,
                                "model__C": c_value,
                                "model__gamma": gamma_value,
                            },
                            "stage1_manual_bank",
                        )
                    )
                    next_index += 1

    rng = np.random.default_rng(RANDOM_STATE)
    if len(candidate_list) > PROFILE["stage1_max_candidates"]:
        sampled_indices = sorted(rng.choice(len(candidate_list), size=PROFILE["stage1_max_candidates"], replace=False))
        stage1_candidates = [candidate_list[idx] for idx in sampled_indices]
    else:
        stage1_candidates = candidate_list

    write_json_atomic(CHECKPOINT_DIR / "stage1_candidates.json", stage1_candidates)
    stage1_results_path = CHECKPOINT_DIR / "stage1_holdout_results.csv"
    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    X_stage1_fit, X_stage1_eval, y_stage1_fit, y_stage1_eval = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    print("Stage 1 candidates:", len(stage1_candidates))
    print("Stage 1 completed:", len(completed_stage1))

    if RUN_STAGE_1:
        for candidate in tqdm([c for c in stage1_candidates if c["candidate_id"] not in completed_stage1], desc="Stage 1 signal feature screening"):
            accuracy, elapsed = evaluate_candidate(X_stage1_fit, y_stage1_fit, X_stage1_eval, y_stage1_eval, candidate)
            row = {
                **candidate_to_row(candidate),
                "stage": "stage1_holdout",
                "holdout_accuracy": accuracy,
                "fit_seconds": elapsed,
            }
            stage1_results = pd.concat([stage1_results, pd.DataFrame([row])], ignore_index=True) if not stage1_results.empty else pd.DataFrame([row])
            stage1_results = stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(stage1_results, stage1_results_path)
            update_manifest(
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_holdout_accuracy": float(stage1_results["holdout_accuracy"].max()),
                }
            )
            gc.collect()

    stage1_results = read_dataframe(stage1_results_path)
    if not stage1_results.empty:
        display(stage1_results.head(15))
        checkpoint_housekeeping("stage1_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SIGNAL_FEATURES_STAGE2 = dedent(
    """
    stage2_fold_path = CHECKPOINT_DIR / "stage2_cv_fold_results.csv"
    stage2_summary_path = CHECKPOINT_DIR / "stage2_cv_summary.csv"

    if stage1_results.empty:
        raise RuntimeError("Stage 1 produced no results.")

    stage2_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage2_shortlist_from_stage1")
        for row in (
            stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True])
            .drop_duplicates("signature")
            .head(PROFILE["stage2_top_k"])
            .to_dict(orient="records")
        )
    ]

    stage2_cv = StratifiedKFold(n_splits=PROFILE["stage2_cv"], shuffle=True, random_state=RANDOM_STATE)
    stage2_splits = list(stage2_cv.split(X_train, y_train))
    fold_df = read_dataframe(stage2_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()

    print("Stage 2 candidates:", len(stage2_candidates))
    print("Stage 2 completed fold-pairs:", len(completed_pairs))

    if RUN_STAGE_2:
        for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(stage2_splits, desc="Stage 2 folds")):
            pending = [candidate for candidate in stage2_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
            for candidate in tqdm(pending, desc=f"Stage 2 fold {fold_idx}", leave=False):
                accuracy, elapsed = evaluate_candidate(
                    X_train[fit_idx],
                    y_train[fit_idx],
                    X_train[eval_idx],
                    y_train[eval_idx],
                    candidate,
                )
                row = {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "fold_idx": fold_idx,
                    "fold_accuracy": accuracy,
                    "fit_seconds": elapsed,
                }
                fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True) if not fold_df.empty else pd.DataFrame([row])
                save_dataframe_atomic(fold_df, stage2_fold_path)
                completed_pairs.add((candidate["candidate_id"], fold_idx))
                gc.collect()

            stage2_summary = refresh_summary_from_folds(stage2_fold_path, stage2_summary_path, stage2_candidates, PROFILE["stage2_cv"], "stage2_cv")
            if not stage2_summary.empty:
                update_manifest(
                    {
                        "stage2_completed_candidates": int(stage2_summary["candidate_id"].nunique()),
                        "stage2_total_candidates": len(stage2_candidates),
                        "stage2_best_cv_accuracy": float(stage2_summary["cv_mean_accuracy"].max()),
                    }
                )

    stage2_summary = read_dataframe(stage2_summary_path)
    if not stage2_summary.empty:
        display(stage2_summary.head(15))
        checkpoint_housekeeping("stage2_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SIGNAL_FEATURES_STAGE3 = dedent(
    """
    stage3_summary = stage2_summary.copy() if not stage2_summary.empty else pd.DataFrame()
    if not stage3_summary.empty:
        display(stage3_summary.head(12))
        checkpoint_housekeeping("stage3_proxy_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


SIGNAL_FEATURES_PLOTS = dedent(
    """
    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        stage1_plot = stage1_results.groupby(["feature__set", "model__family"], as_index=False)["holdout_accuracy"].max()
        sns.barplot(data=stage1_plot, x="feature__set", y="holdout_accuracy", hue="model__family")
        plt.title("Best holdout accuracy by feature set and model family")
        save_current_figure("stage1_feature_family.png")
        display(stage1_plot)

    if not stage2_summary.empty:
        plt.figure(figsize=(12, 6))
        top_df = stage2_summary.head(15).copy()
        sns.barplot(data=top_df, x="candidate_id", y="cv_mean_accuracy", hue="model__family")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top signal feature candidates")
        save_current_figure("top_candidates.png")
    """
).strip()


SIGNAL_FEATURES_FINAL = dedent(
    """
    final_summary_df = stage2_summary if not stage2_summary.empty else stage1_results
    if final_summary_df.empty:
        raise RuntimeError("No final candidate table is available.")

    best_row = final_summary_df.iloc[0].to_dict()
    best_candidate = candidate_record(best_row["candidate_id"], json.loads(best_row["params_json"]), "final_selection")

    X_fit_matrix, X_eval_matrix = build_matrix(X_train, X_valid, best_candidate)
    final_model = build_model(best_candidate, probability=False)
    final_model.fit(X_fit_matrix, y_train)
    valid_predictions = final_model.predict(X_eval_matrix)
    validation_accuracy = float(accuracy_score(y_valid, valid_predictions))

    cm = confusion_matrix(y_valid, valid_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Validation confusion matrix - accuracy={validation_accuracy:.4f}")
    save_current_figure("validation_confusion_matrix.png")

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_1 = np.zeros(len(X_full), dtype=np.float32)
    test_prob_1 = np.zeros(len(X_test_full), dtype=np.float32)
    fold_records = []

    for fold_idx, (fit_idx, eval_idx) in enumerate(tqdm(oof_cv.split(X_full, y_full), desc="OOF for stacking")):
        X_fold_fit, X_fold_eval = build_matrix(X_full[fit_idx], X_full[eval_idx], best_candidate)
        X_fold_fit_again, X_fold_test = build_matrix(X_full[fit_idx], X_test_full, best_candidate)
        if X_fold_fit.shape != X_fold_fit_again.shape:
            raise RuntimeError("Unexpected feature matrix mismatch.")
        fold_model = build_model(best_candidate, probability=(best_candidate["params"]["model__family"] == "svm"))
        fold_model.fit(X_fold_fit, y_full[fit_idx])
        if best_candidate["params"]["model__family"] == "knn":
            oof_proba = fold_model.predict_proba(X_fold_eval)[:, 1]
            test_proba = fold_model.predict_proba(X_fold_test)[:, 1]
        else:
            oof_proba = fold_model.predict_proba(X_fold_eval)[:, 1]
            test_proba = fold_model.predict_proba(X_fold_test)[:, 1]
        oof_prob_1[eval_idx] = oof_proba.astype(np.float32)
        test_prob_1 += test_proba.astype(np.float32) / oof_cv.n_splits
        fold_records.append(
            {
                "fold_idx": fold_idx,
                "fold_accuracy": float(accuracy_score(y_full[eval_idx], (oof_proba >= 0.5).astype(int))),
            }
        )
        gc.collect()

    X_full_matrix, X_test_matrix = build_matrix(X_full, X_test_full, best_candidate)
    final_model_full = build_model(best_candidate, probability=False)
    final_model_full.fit(X_full_matrix, y_full)
    final_test_pred = final_model_full.predict(X_test_matrix).astype(int)

    oof_df = pd.DataFrame(
        {
            "id": train_ids,
            "y_true": y_full.astype(int),
            "prob_1": oof_prob_1,
            "pred": (oof_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    test_prob_df = pd.DataFrame(
        {
            "id": test_ids,
            "prob_1": test_prob_1,
            "pred": (test_prob_1 >= 0.5).astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    submission_df = sample_df.copy()
    submission_df["class"] = final_test_pred.astype(int)

    oof_path = PERSIST_ROOT / "oof_probabilities.csv"
    test_prob_path = PERSIST_ROOT / "test_probabilities.csv"
    fold_summary_path = PERSIST_ROOT / "oof_fold_summary.csv"
    submission_path = SUBMISSION_DIR / "challenge_10_signal_features_colab_ultra_submission.csv"
    summary_path = PERSIST_ROOT / "summary.json"

    save_dataframe_atomic(oof_df, oof_path)
    save_dataframe_atomic(test_prob_df, test_prob_path)
    save_dataframe_atomic(pd.DataFrame(fold_records), fold_summary_path)
    submission_df.to_csv(submission_path, index=False)

    summary_payload = {
        "model_name": "Signal feature engineering model",
        "model_key": "signal_features",
        "notebook_slug": NOTEBOOK_SLUG,
        "strategy": "colab_ultra_signal_features",
        "search_profile": SEARCH_PROFILE,
        "best_stage": str(best_row.get("stage", "unknown")),
        "best_params": best_candidate["params"],
        "validation_accuracy": validation_accuracy,
        "validation_confusion_matrix": cm.tolist(),
        "oof_accuracy": float(accuracy_score(y_full, (oof_prob_1 >= 0.5).astype(int))),
        "oof_path": str(oof_path),
        "test_probability_path": str(test_prob_path),
        "submission_path": str(submission_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "persist_root": str(PERSIST_ROOT),
    }
    write_json_atomic(summary_path, summary_payload)
    checkpoint_housekeeping("final_model_complete", refresh_bundle=True, include_data_in_bundle=False)

    print("Best params:", json.dumps(best_candidate["params"], indent=2))
    print("Validation accuracy:", validation_accuracy)
    print("OOF accuracy:", summary_payload["oof_accuracy"])
    print("Submission path:", submission_path)
    """
).strip()


SIGNAL_FEATURES_OUTRO = dedent(
    """
    ## Notes

    Si esta notebook supera a los pipelines con `raw features`, la evidencia es fuerte:

    - el orden de `V1 ... V200` si contiene estructura util de señal
    - el challenge se beneficia mas de `feature engineering` que de seguir moviendo hiperparametros
    """
).strip()


STACKING_INTRO = dedent(
    """
    # Challenge 11 Colab Ultra: stacking and blending

    Esta notebook no entrena modelos base desde cero.

    Su objetivo es combinar las salidas de las otras variantes avanzadas:

    - `KNN cleaning`
    - `SVM preprocessing`
    - `signal features`

    Para ello consume los artefactos:

    - `oof_probabilities.csv`
    - `test_probabilities.csv`

    que quedan dentro del `output/` de cada notebook anterior.
    """
).strip()


STACKING_IMPORTS = common_imports(
    "challenge_11_stacking_colab_ultra",
    dedent(
        """
        from sklearn.linear_model import LogisticRegression
        """
    ).strip(),
)


STACKING_RUNTIME = dedent(
    """
    SEARCH_PROFILE = "aggressive"
    RUN_DISCOVERY = True
    RUN_SEARCH = True
    TRAIN_FINAL_MODEL = True

    META_C_VALUES = [0.25, 1.0, 4.0, 16.0]
    print({"python": platform.python_version(), "sklearn": sklearn.__version__})
    """
).strip()


STACKING_HELPERS = dedent(
    """
    def discover_probability_artifacts() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        oof_paths = sorted(OUTPUT_ROOT.glob("*/oof_probabilities.csv"))
        test_paths = sorted(OUTPUT_ROOT.glob("*/test_probabilities.csv"))

        oof_map = {}
        test_map = {}
        for path in oof_paths:
            df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            source_name = str(df["source_model"].iloc[0]) if "source_model" in df.columns else path.parent.name
            oof_map[source_name] = df
        for path in test_paths:
            df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            source_name = str(df["source_model"].iloc[0]) if "source_model" in df.columns else path.parent.name
            test_map[source_name] = df
        return oof_map, test_map


    def align_artifacts(oof_map: dict[str, pd.DataFrame], test_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_models = sorted(set(oof_map) & set(test_map))
        if len(common_models) < 2:
            raise RuntimeError("At least two base models with both OOF and test probabilities are required for stacking.")

        base_oof = oof_map[common_models[0]][["id", "y_true"]].copy()
        base_test = test_map[common_models[0]][["id"]].copy()
        for model_name in common_models:
            model_oof = oof_map[model_name][["id", "prob_1"]].rename(columns={"prob_1": model_name})
            model_test = test_map[model_name][["id", "prob_1"]].rename(columns={"prob_1": model_name})
            base_oof = base_oof.merge(model_oof, on="id", how="inner")
            base_test = base_test.merge(model_test, on="id", how="inner")
        return base_oof, base_test


    def evaluate_average_subset(meta_df: pd.DataFrame, subset: list[str]) -> dict:
        probs = meta_df[subset].mean(axis=1).to_numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "meta_family": "average",
            "base_models": subset,
            "C": None,
            "cv_accuracy": float(accuracy_score(meta_df["y_true"], preds)),
        }


    def evaluate_logreg_subset(meta_df: pd.DataFrame, subset: list[str], c_value: float) -> dict:
        X_meta = meta_df[subset].to_numpy(dtype=np.float32)
        y_meta = meta_df["y_true"].to_numpy(dtype=np.int8)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_prob = np.zeros(len(meta_df), dtype=np.float32)
        for fit_idx, eval_idx in cv.split(X_meta, y_meta):
            model = LogisticRegression(C=c_value, max_iter=2000)
            model.fit(X_meta[fit_idx], y_meta[fit_idx])
            oof_prob[eval_idx] = model.predict_proba(X_meta[eval_idx])[:, 1].astype(np.float32)
        preds = (oof_prob >= 0.5).astype(int)
        return {
            "meta_family": "logreg",
            "base_models": subset,
            "C": c_value,
            "cv_accuracy": float(accuracy_score(y_meta, preds)),
        }
    """
).strip()


STACKING_STAGE1 = dedent(
    """
    if RUN_DISCOVERY:
        oof_map, test_map = discover_probability_artifacts()
        print("Discovered OOF artifacts:", sorted(oof_map))
        print("Discovered test artifacts:", sorted(test_map))
    else:
        oof_map, test_map = {}, {}

    meta_oof_df, meta_test_df = align_artifacts(oof_map, test_map)
    print("Meta OOF shape:", meta_oof_df.shape)
    print("Meta test shape:", meta_test_df.shape)
    display(meta_oof_df.head())

    corr_df = meta_oof_df.drop(columns=["id", "y_true"]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Correlation of base-model OOF probabilities")
    save_current_figure("oof_probability_correlation.png")
    checkpoint_housekeeping("stage1_artifact_discovery_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STACKING_STAGE2 = dedent(
    """
    search_results_path = CHECKPOINT_DIR / "stacking_search_results.csv"
    search_df = read_dataframe(search_results_path)
    completed_signatures = set(search_df["signature"]) if not search_df.empty else set()

    base_model_names = [column for column in meta_oof_df.columns if column not in {"id", "y_true"}]
    search_candidates = []
    next_index = 0
    for subset_size in range(2, len(base_model_names) + 1):
        for subset in itertools.combinations(base_model_names, subset_size):
            subset_list = list(subset)
            avg_signature = json.dumps({"meta_family": "average", "base_models": subset_list}, sort_keys=True)
            search_candidates.append({"candidate_id": f"blend_{next_index:03d}", "signature": avg_signature, "meta_family": "average", "base_models": subset_list, "C": None})
            next_index += 1
            for c_value in META_C_VALUES:
                signature = json.dumps({"meta_family": "logreg", "base_models": subset_list, "C": c_value}, sort_keys=True)
                search_candidates.append({"candidate_id": f"blend_{next_index:03d}", "signature": signature, "meta_family": "logreg", "base_models": subset_list, "C": c_value})
                next_index += 1

    print("Stacking candidates:", len(search_candidates))
    print("Already completed:", len(completed_signatures))

    if RUN_SEARCH:
        pending = [candidate for candidate in search_candidates if candidate["signature"] not in completed_signatures]
        for candidate in tqdm(pending, desc="Stacking candidates"):
            if candidate["meta_family"] == "average":
                result = evaluate_average_subset(meta_oof_df, candidate["base_models"])
            else:
                result = evaluate_logreg_subset(meta_oof_df, candidate["base_models"], float(candidate["C"]))
            row = {
                "candidate_id": candidate["candidate_id"],
                "signature": candidate["signature"],
                "meta_family": result["meta_family"],
                "base_models_json": json.dumps(result["base_models"]),
                "C": result["C"],
                "cv_accuracy": result["cv_accuracy"],
            }
            search_df = pd.concat([search_df, pd.DataFrame([row])], ignore_index=True) if not search_df.empty else pd.DataFrame([row])
            search_df = search_df.sort_values(["cv_accuracy"], ascending=[False]).reset_index(drop=True)
            save_dataframe_atomic(search_df, search_results_path)

    search_df = read_dataframe(search_results_path)
    display(search_df.head(20))
    checkpoint_housekeeping("stage2_stacking_search_complete", refresh_bundle=True, include_data_in_bundle=False)
    """
).strip()


STACKING_STAGE3 = dedent(
    """
    stacking_summary = search_df.copy()
    """
).strip()


STACKING_PLOTS = dedent(
    """
    if not search_df.empty:
        plt.figure(figsize=(12, 6))
        plot_df = search_df.head(20).copy()
        sns.barplot(data=plot_df, x="candidate_id", y="cv_accuracy", hue="meta_family")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top stacking candidates")
        save_current_figure("top_stacking_candidates.png")
    """
).strip()


STACKING_FINAL = dedent(
    """
    if search_df.empty:
        raise RuntimeError("No stacking search results are available.")

    best_row = search_df.iloc[0].to_dict()
    base_models = json.loads(best_row["base_models_json"])
    meta_family = best_row["meta_family"]

    X_meta_train = meta_oof_df[base_models].to_numpy(dtype=np.float32)
    y_meta = meta_oof_df["y_true"].to_numpy(dtype=np.int8)
    X_meta_test = meta_test_df[base_models].to_numpy(dtype=np.float32)

    if meta_family == "average":
        train_prob = X_meta_train.mean(axis=1)
        test_prob = X_meta_test.mean(axis=1)
    else:
        meta_model = LogisticRegression(C=float(best_row["C"]), max_iter=2000)
        meta_model.fit(X_meta_train, y_meta)
        train_prob = meta_model.predict_proba(X_meta_train)[:, 1]
        test_prob = meta_model.predict_proba(X_meta_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_meta, train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Meta-model training confusion matrix - accuracy={accuracy_score(y_meta, train_pred):.4f}")
    save_current_figure("meta_training_confusion_matrix.png")

    submission_df = sample_df.copy()
    submission_df["class"] = test_pred.astype(int)
    submission_path = SUBMISSION_DIR / "challenge_11_stacking_colab_ultra_submission.csv"
    submission_df.to_csv(submission_path, index=False)

    summary_payload = {
        "model_name": "Stacking / blending meta-model",
        "model_key": "stacking",
        "notebook_slug": NOTEBOOK_SLUG,
        "strategy": "colab_ultra_stacking",
        "best_meta_family": meta_family,
        "best_base_models": base_models,
        "best_C": None if pd.isna(best_row["C"]) else float(best_row["C"]),
        "meta_training_accuracy": float(accuracy_score(y_meta, train_pred)),
        "submission_path": str(submission_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "persist_root": str(PERSIST_ROOT),
    }
    write_json_atomic(PERSIST_ROOT / "summary.json", summary_payload)
    checkpoint_housekeeping("final_model_complete", refresh_bundle=True, include_data_in_bundle=False)

    print("Best stacking candidate:", json.dumps(summary_payload, indent=2))
    print("Submission path:", submission_path)
    """
).strip()


STACKING_OUTRO = dedent(
    """
    ## Notes

    Esta notebook debe ejecutarse despues de que al menos dos notebooks base hayan generado:

    - `oof_probabilities.csv`
    - `test_probabilities.csv`

    Lo normal es restaurar los ZIP de esas corridas dentro del mismo workspace antes de correr el stacking.
    """
).strip()


def md_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_strategy_notebook(spec: StrategySpec, *, allow_multi_zip: bool = False) -> nbf.NotebookNode:
    setup_code, status_code, restore_code = common_setup(spec.workspace_name, allow_multi_zip=allow_multi_zip)
    cells = [
        md_cell(spec.intro),
        md_cell("## 0. Imports and global constants"),
        code_cell(spec.imports_code),
        md_cell("## 1. Create the Colab workspace"),
        code_cell(setup_code),
        md_cell("## 2. Inspect the workspace and expected files"),
        code_cell(status_code),
        md_cell("## 3. Optional: upload the CSV files manually"),
        code_cell(COMMON_UPLOAD_DATA),
        md_cell("## 4. Optional: restore resume bundles"),
        code_cell(restore_code),
        md_cell("## 5. Checkpoint helpers and reusable utilities"),
        code_cell(common_helpers(f"{spec.notebook_slug}_resume.zip")),
        md_cell("## 6. Strategy-specific helpers"),
        code_cell(spec.strategy_helpers_code),
        md_cell("## 7. Validate that the required CSV files are present"),
        code_cell(COMMON_VALIDATE_DATA),
        md_cell("## 8. Runtime inspection and search budget"),
        code_cell(spec.runtime_code),
        md_cell("## 9. Load the challenge data"),
        code_cell(COMMON_LOAD_DATA),
        md_cell("## 10. Stage 1"),
        code_cell(spec.stage1_code),
        md_cell("## 11. Stage 2"),
        code_cell(spec.stage2_code),
        md_cell("## 12. Stage 3"),
        code_cell(spec.stage3_code),
        md_cell("## 13. Diagnostic plots"),
        code_cell(spec.plots_code),
        md_cell("## 14. Final model, OOF artifacts and submission"),
        code_cell(spec.final_code),
        md_cell("## 15. Optional: export and download a manual resume bundle"),
        code_cell(COMMON_EXPORT_BUNDLE),
        md_cell(spec.outro),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return notebook


def build_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            folder_name="colab_knn_cleaning_ultra",
            notebook_name="Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb",
            notebook_title="Challenge 08 Colab Ultra: KNN with cleaning",
            notebook_slug="challenge_08_knn_cleaning_colab_ultra",
            workspace_name="challenge_knn_cleaning_ultra_workspace",
            readme_title="KNN Cleaning Ultra for Colab",
            readme_text=dedent(
                """
                Esta carpeta implementa la linea `KNN + limpieza de instancias`.

                El objetivo es mejorar KNN atacando directamente puntos ruidosos o ambiguos, usando tecnicas compatibles con la regla de preprocesamiento libre.

                Incluye:

                - screening por etapas
                - checkpoints
                - ZIP de reanudacion
                - export de `OOF probabilities` y `test probabilities` para stacking

                Flujo recomendado:

                1. subir solo el notebook a Google Colab
                2. usar runtime CPU
                3. ejecutar las celdas iniciales
                4. en `Optional: upload the CSV files manually` cambiar `UPLOAD_DATA_FILES = True`
                5. subir `training.csv`, `test.csv` y `sample.csv`
                6. dejar `SEARCH_PROFILE = "balanced"` si estas en Colab free o `aggressive` si tienes mas margen
                7. correr hasta terminar `Stage 2`
                8. exportar y descargar el ZIP de reanudacion
                9. si la sesion se corta, abrir una nueva, restaurar el ZIP y continuar con `Stage 3` y la fase final

                Artefactos finales esperados:

                - `summary.json`
                - `oof_probabilities.csv`
                - `test_probabilities.csv`
                - `submissions/challenge_08_knn_cleaning_colab_ultra_submission.csv`
                """
            ).strip(),
            intro=KNN_CLEANING_INTRO,
            imports_code=KNN_CLEANING_IMPORTS,
            runtime_code=KNN_CLEANING_RUNTIME,
            strategy_helpers_code=KNN_CLEANING_HELPERS,
            stage1_code=KNN_CLEANING_STAGE1,
            stage2_code=KNN_CLEANING_STAGE2,
            stage3_code=KNN_CLEANING_STAGE3,
            plots_code=KNN_CLEANING_PLOTS,
            final_code=KNN_CLEANING_FINAL,
            outro=KNN_CLEANING_OUTRO,
        ),
        StrategySpec(
            folder_name="colab_svm_preprocessing_ultra",
            notebook_name="Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb",
            notebook_title="Challenge 09 Colab Ultra: SVM with preprocessing",
            notebook_slug="challenge_09_svm_preprocessing_colab_ultra",
            workspace_name="challenge_svm_preprocessing_ultra_workspace",
            readme_title="SVM Preprocessing Ultra for Colab",
            readme_text=dedent(
                """
                Esta carpeta implementa la linea `SVM + preprocesamiento`.

                La idea es dejar fija la region buena de `SVM` y buscar mejoras via:

                - escalado
                - transformaciones de distribucion
                - limpieza suave de instancias

                Tambien exporta artefactos de probabilidades para stacking.

                Flujo recomendado:

                1. subir solo el notebook a Colab
                2. usar runtime CPU
                3. ejecutar las celdas iniciales
                4. en la celda de upload poner `UPLOAD_DATA_FILES = True`
                5. subir `training.csv`, `test.csv` y `sample.csv`
                6. dejar `SEARCH_PROFILE = "balanced"` en Colab free si quieres una corrida inicial mas segura
                7. correr `Stage 1` y `Stage 2`
                8. descargar el ZIP exportado antes de cerrar la sesion
                9. restaurar ese ZIP si luego quieres completar `Stage 3` y la fase final

                Artefactos finales esperados:

                - `summary.json`
                - `oof_probabilities.csv`
                - `test_probabilities.csv`
                - `submissions/challenge_09_svm_preprocessing_colab_ultra_submission.csv`
                """
            ).strip(),
            intro=SVM_PREP_INTRO,
            imports_code=SVM_PREP_IMPORTS,
            runtime_code=SVM_PREP_RUNTIME,
            strategy_helpers_code=SVM_PREP_HELPERS,
            stage1_code=SVM_PREP_STAGE1,
            stage2_code=SVM_PREP_STAGE2,
            stage3_code=SVM_PREP_STAGE3,
            plots_code=SVM_PREP_PLOTS,
            final_code=SVM_PREP_FINAL,
            outro=SVM_PREP_OUTRO,
        ),
        StrategySpec(
            folder_name="colab_signal_features_ultra",
            notebook_name="Challenge_10_Signal_Features_Colab_Ultra.ipynb",
            notebook_title="Challenge 10 Colab Ultra: signal feature engineering",
            notebook_slug="challenge_10_signal_features_colab_ultra",
            workspace_name="challenge_signal_features_ultra_workspace",
            readme_title="Signal Features Ultra for Colab",
            readme_text=dedent(
                """
                Esta carpeta implementa la linea `signal feature engineering`.

                El supuesto central es que `V1 ... V200` si representan una secuencia ordenada de señal.

                La notebook construye:

                - features temporales
                - features frecuenciales
                - features por segmentos
                - una variante hibrida con `raw PCA`

                y compara `KNN` y `SVM` sobre esas representaciones.

                Flujo recomendado:

                1. subir solo el notebook a Colab
                2. usar runtime CPU
                3. ejecutar las celdas iniciales
                4. en la celda de upload poner `UPLOAD_DATA_FILES = True`
                5. subir `training.csv`, `test.csv` y `sample.csv`
                6. ejecutar `Stage 1` para identificar si las features de senal muestran ventaja real
                7. continuar con `Stage 2` para confirmar estabilidad
                8. exportar y descargar el ZIP al terminar

                Artefactos finales esperados:

                - `summary.json`
                - `oof_probabilities.csv`
                - `test_probabilities.csv`
                - `submissions/challenge_10_signal_features_colab_ultra_submission.csv`
                """
            ).strip(),
            intro=SIGNAL_FEATURES_INTRO,
            imports_code=SIGNAL_FEATURES_IMPORTS,
            runtime_code=SIGNAL_FEATURES_RUNTIME,
            strategy_helpers_code=SIGNAL_FEATURES_HELPERS,
            stage1_code=SIGNAL_FEATURES_STAGE1,
            stage2_code=SIGNAL_FEATURES_STAGE2,
            stage3_code=SIGNAL_FEATURES_STAGE3,
            plots_code=SIGNAL_FEATURES_PLOTS,
            final_code=SIGNAL_FEATURES_FINAL,
            outro=SIGNAL_FEATURES_OUTRO,
        ),
        StrategySpec(
            folder_name="colab_stacking_ultra",
            notebook_name="Challenge_11_Stacking_Colab_Ultra.ipynb",
            notebook_title="Challenge 11 Colab Ultra: stacking and blending",
            notebook_slug="challenge_11_stacking_colab_ultra",
            workspace_name="challenge_stacking_ultra_workspace",
            readme_title="Stacking Ultra for Colab",
            readme_text=dedent(
                """
                Esta carpeta implementa la fase final de blending/stacking.

                No entrena modelos base desde cero. Consume los artefactos exportados por:

                - KNN cleaning
                - SVM preprocessing
                - signal features

                y busca la mejor combinacion con:

                - promedio simple
                - `LogisticRegression` como meta-modelo

                Flujo recomendado:

                1. ejecutar antes al menos dos notebooks base y descargar sus ZIP exportados
                2. subir solo este notebook a Colab
                3. ejecutar las celdas iniciales
                4. en `Optional: restore resume bundles` cambiar `RESTORE_RESUME_BUNDLES = True`
                5. subir dos o mas ZIPs de reanudacion provenientes de los modelos base
                6. verificar que el notebook descubra `oof_probabilities.csv` y `test_probabilities.csv`
                7. correr la busqueda de blending y generar la submission final

                El stacking no sirve si los modelos base no dejaron estos artefactos:

                - `oof_probabilities.csv`
                - `test_probabilities.csv`
                """
            ).strip(),
            intro=STACKING_INTRO,
            imports_code=STACKING_IMPORTS,
            runtime_code=STACKING_RUNTIME,
            strategy_helpers_code=STACKING_HELPERS,
            stage1_code=STACKING_STAGE1,
            stage2_code=STACKING_STAGE2,
            stage3_code=STACKING_STAGE3,
            plots_code=STACKING_PLOTS,
            final_code=STACKING_FINAL,
            outro=STACKING_OUTRO,
        ),
    ]


def write_bundle(spec: StrategySpec) -> None:
    target_dir = ROOT / spec.folder_name
    notebook_path = target_dir / spec.notebook_name
    readme_path = target_dir / "README.md"
    requirements_path = target_dir / "requirements.txt"
    workspace_template_dir = target_dir / "workspace_template"
    workspace_readme_path = workspace_template_dir / "README.md"

    target_dir.mkdir(parents=True, exist_ok=True)
    workspace_template_dir.mkdir(parents=True, exist_ok=True)

    allow_multi_zip = spec.folder_name == "colab_stacking_ultra"
    notebook = build_strategy_notebook(spec, allow_multi_zip=allow_multi_zip)
    nbf.write(notebook, notebook_path)
    readme_path.write_text(f"# {spec.readme_title}\n\n{spec.readme_text}\n")
    requirements_path.write_text(REQUIREMENTS_TEXT)
    workspace_readme_path.write_text(WORKSPACE_TEMPLATE_TEXT)
    print(f"Created {notebook_path}")


def main() -> None:
    for spec in build_specs():
        write_bundle(spec)


if __name__ == "__main__":
    main()
