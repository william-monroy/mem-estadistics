from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "colab_final_stacking"
NOTEBOOK_PATH = TARGET_DIR / "Challenge_12_Final_Stacking_Colab.ipynb"
README_PATH = TARGET_DIR / "README.md"
REQUIREMENTS_PATH = TARGET_DIR / "requirements.txt"
WORKSPACE_TEMPLATE_DIR = TARGET_DIR / "workspace_template"
WORKSPACE_TEMPLATE_PATH = WORKSPACE_TEMPLATE_DIR / "README.md"


INTRO = dedent(
    """
    # Challenge 12 Final Colab: focused stacking for Kaggle

    Esta notebook final no vuelve a entrenar modelos base.

    Esta pensada para tu situacion actual:

    - `signal_features` es el modelo fuerte
    - `knn_cleaning` aporta diversidad real
    - `svm_preprocessing` es opcional

    ## Objetivo

    Combinar los artefactos ya generados por las corridas base:

    - `oof_probabilities.csv`
    - `test_probabilities.csv`
    - `summary.json` si existe

    y buscar una combinacion final mas fuerte mediante:

    1. `weighted average` con busqueda de pesos
    2. ajuste de umbral (`threshold tuning`)
    3. `LogisticRegression` como meta-modelo

    ## Diseno operativo

    - pensado para una sesion limpia de Google Colab
    - no requiere `training.csv`, `test.csv` ni `sample.csv`
    - acepta multiples ZIPs de corridas previas
    - tambien puede reanudarse si guardas su bundle final
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
    import itertools
    import json
    import platform
    import re
    import warnings
    import zipfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    try:
        import google.colab  # type: ignore
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        from google.colab import files  # type: ignore
    else:
        files = None

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.bbox"] = "tight"

    RANDOM_STATE = 301655
    NOTEBOOK_SLUG = "challenge_12_final_stacking_colab"
    """
).strip()


SETUP = dedent(
    """
    if IN_COLAB:
        WORKSPACE_ROOT = Path("/content/challenge_final_stacking_workspace")
    else:
        cwd = Path.cwd().resolve()
        candidate_roots = []
        for base in [cwd, *cwd.parents]:
            candidate_roots.extend(
                [
                    base / "challenge" / "results-pre-stack",
                    base / "results-pre-stack",
                ]
            )
        WORKSPACE_ROOT = next((path for path in candidate_roots if path.exists()), cwd / "challenge_final_stacking_workspace")

    INPUT_BUNDLE_DIR = WORKSPACE_ROOT / "input_bundles"
    OUTPUT_ROOT = WORKSPACE_ROOT / "output"
    PERSIST_ROOT = OUTPUT_ROOT / NOTEBOOK_SLUG
    CHECKPOINT_DIR = PERSIST_ROOT / "checkpoints"
    SUBMISSION_DIR = WORKSPACE_ROOT / "submissions"
    EXPORT_DIR = WORKSPACE_ROOT / "exports"
    ARTIFACT_SEARCH_ROOT = WORKSPACE_ROOT

    for path in [WORKSPACE_ROOT, INPUT_BUNDLE_DIR, OUTPUT_ROOT, PERSIST_ROOT, CHECKPOINT_DIR, SUBMISSION_DIR, EXPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    print("IN_COLAB:", IN_COLAB)
    print("WORKSPACE_ROOT:", WORKSPACE_ROOT)
    print("ARTIFACT_SEARCH_ROOT:", ARTIFACT_SEARCH_ROOT)
    print("PERSIST_ROOT:", PERSIST_ROOT)
    """
).strip()


STATUS = dedent(
    """
    print("Workspace directories:")
    for path in [WORKSPACE_ROOT, INPUT_BUNDLE_DIR, OUTPUT_ROOT, PERSIST_ROOT, CHECKPOINT_DIR, SUBMISSION_DIR, EXPORT_DIR]:
        print("-", path)

    existing_oof = sorted(path for path in ARTIFACT_SEARCH_ROOT.rglob("oof_probabilities.csv") if PERSIST_ROOT not in path.parents)
    existing_test = sorted(path for path in ARTIFACT_SEARCH_ROOT.rglob("test_probabilities.csv") if PERSIST_ROOT not in path.parents)

    print("\\nAlready discovered files before upload:")
    print("- oof_probabilities.csv:", len(existing_oof))
    print("- test_probabilities.csv:", len(existing_test))
    """
).strip()


UPLOAD_BUNDLES = dedent(
    """
    UPLOAD_RESULT_BUNDLES = False

    if UPLOAD_RESULT_BUNDLES:
        if not IN_COLAB:
            raise RuntimeError("This upload helper is intended for Google Colab.")

        uploaded = files.upload()
        zip_names = [Path(name).name for name in uploaded if str(name).lower().endswith(".zip")]
        if not zip_names:
            raise ValueError("Upload one or more ZIP bundles.")

        for bundle_name in zip_names:
            bundle_path = INPUT_BUNDLE_DIR / bundle_name
            bundle_path.write_bytes(uploaded[bundle_name])
            with zipfile.ZipFile(bundle_path, "r") as zip_file:
                zip_file.extractall(WORKSPACE_ROOT)
            print("Restored bundle:", bundle_path)
    else:
        print("Set UPLOAD_RESULT_BUNDLES = True to upload ZIP bundles produced by the base-model notebooks.")
    """
).strip()


HELPERS = dedent(
    """
    DEFAULT_BUNDLE_NAME = "challenge_12_final_stacking_colab_resume.zip"


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


    def create_resume_bundle(bundle_name: str = DEFAULT_BUNDLE_NAME) -> Path:
        bundle_path = EXPORT_DIR / bundle_name
        if bundle_path.exists():
            bundle_path.unlink()

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for root_path in [OUTPUT_ROOT, SUBMISSION_DIR]:
                if not root_path.exists():
                    continue
                for nested in root_path.rglob("*"):
                    if nested.is_dir():
                        continue
                    relative_path = nested.relative_to(WORKSPACE_ROOT)
                    zip_file.write(nested, arcname=str(relative_path))

        return bundle_path


    def safe_slug(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "model"


    def make_unique_name(base_name: str, used: set[str]) -> str:
        candidate = safe_slug(base_name)
        if candidate not in used:
            used.add(candidate)
            return candidate
        index = 2
        while f"{candidate}_{index}" in used:
            index += 1
        unique = f"{candidate}_{index}"
        used.add(unique)
        return unique


    def load_summary_if_present(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}


    def discover_probability_artifacts() -> list[dict]:
        records = []
        used_names = set()
        for oof_path in sorted(ARTIFACT_SEARCH_ROOT.rglob("oof_probabilities.csv")):
            if PERSIST_ROOT in oof_path.parents:
                continue

            model_root = oof_path.parent
            test_path = model_root / "test_probabilities.csv"
            summary_path = model_root / "summary.json"
            if not test_path.exists():
                continue

            oof_df = pd.read_csv(oof_path).sort_values("id").reset_index(drop=True)
            test_df = pd.read_csv(test_path).sort_values("id").reset_index(drop=True)
            if not {"id", "prob_1", "y_true"}.issubset(oof_df.columns):
                continue
            if not {"id", "prob_1"}.issubset(test_df.columns):
                continue

            summary = load_summary_if_present(summary_path)
            if "source_model" in oof_df.columns and not oof_df["source_model"].empty:
                source_name = str(oof_df["source_model"].iloc[0])
            else:
                source_name = summary.get("model_key") or summary.get("notebook_slug") or model_root.name

            model_name = make_unique_name(source_name, used_names)
            records.append(
                {
                    "model_name": model_name,
                    "model_root": model_root,
                    "summary_path": summary_path,
                    "summary": summary,
                    "oof_path": oof_path,
                    "test_path": test_path,
                    "oof_df": oof_df,
                    "test_df": test_df,
                }
            )

        return records


    def align_artifacts(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(records) < 2:
            raise RuntimeError("At least two base models are required for final stacking.")

        meta_oof_df = records[0]["oof_df"][["id", "y_true"]].copy()
        meta_test_df = records[0]["test_df"][["id"]].copy()

        for record in records:
            model_name = record["model_name"]
            oof_part = record["oof_df"][["id", "prob_1"]].rename(columns={"prob_1": model_name})
            test_part = record["test_df"][["id", "prob_1"]].rename(columns={"prob_1": model_name})
            meta_oof_df = meta_oof_df.merge(oof_part, on="id", how="inner")
            meta_test_df = meta_test_df.merge(test_part, on="id", how="inner")

        meta_oof_df = meta_oof_df.sort_values("id").reset_index(drop=True)
        meta_test_df = meta_test_df.sort_values("id").reset_index(drop=True)
        return meta_oof_df, meta_test_df


    def base_model_summary_table(records: list[dict], meta_oof_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for record in records:
            model_name = record["model_name"]
            probs = meta_oof_df[model_name].to_numpy(dtype=np.float64)
            preds = (probs >= 0.5).astype(int)
            rows.append(
                {
                    "model_name": model_name,
                    "oof_accuracy": float(accuracy_score(meta_oof_df["y_true"], preds)),
                    "positive_rate": float(preds.mean()),
                    "oof_path": str(record["oof_path"]),
                    "test_path": str(record["test_path"]),
                    "summary_path": str(record["summary_path"]) if record["summary_path"].exists() else "",
                    "model_name_raw": record["summary"].get("model_name", ""),
                    "best_stage": record["summary"].get("best_stage", ""),
                }
            )
        return pd.DataFrame(rows).sort_values(["oof_accuracy", "model_name"], ascending=[False, True]).reset_index(drop=True)


    def pairwise_disagreement_table(meta_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
        rows = []
        for left_name, right_name in itertools.combinations(model_names, 2):
            left_pred = (meta_df[left_name] >= 0.5).astype(int)
            right_pred = (meta_df[right_name] >= 0.5).astype(int)
            rows.append(
                {
                    "left_model": left_name,
                    "right_model": right_name,
                    "disagreement_rate": float((left_pred != right_pred).mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("disagreement_rate", ascending=False).reset_index(drop=True)


    def generate_weight_vectors(size: int, step: float) -> list[list[float]]:
        grid = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 6)
        if size == 2:
            return [[float(w), float(round(1.0 - w, 6))] for w in grid]
        if size == 3:
            vectors = []
            for w1 in grid:
                for w2 in grid:
                    w3 = round(1.0 - float(w1) - float(w2), 6)
                    if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                        continue
                    if abs(float(w1) + float(w2) + w3 - 1.0) > 1e-6:
                        continue
                    vectors.append([float(w1), float(w2), float(w3)])
            return vectors
        raise ValueError("Only subset sizes 2 and 3 are supported for weighted averages.")


    def find_best_threshold(y_true: np.ndarray, prob_1: np.ndarray, threshold_grid: np.ndarray) -> tuple[float, float]:
        best_threshold = 0.5
        best_accuracy = -1.0
        for threshold in threshold_grid:
            preds = (prob_1 >= float(threshold)).astype(int)
            accuracy = accuracy_score(y_true, preds)
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = float(accuracy)
                best_threshold = float(threshold)
        return best_threshold, best_accuracy


    def weighted_probabilities(meta_df: pd.DataFrame, subset: list[str], weights: list[float]) -> np.ndarray:
        matrix = meta_df[subset].to_numpy(dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        return (matrix * weight_array[None, :]).sum(axis=1)


    def evaluate_weighted_candidate(meta_df: pd.DataFrame, subset: list[str], weights: list[float], threshold_grid: np.ndarray) -> dict:
        prob_1 = weighted_probabilities(meta_df, subset, weights)
        threshold, accuracy = find_best_threshold(meta_df["y_true"].to_numpy(dtype=np.int8), prob_1, threshold_grid)
        return {
            "meta_family": "weighted_average",
            "base_models": subset,
            "weights": weights,
            "C": None,
            "best_threshold": threshold,
            "meta_oof_accuracy": accuracy,
        }


    def generate_logreg_oof_probabilities(meta_df: pd.DataFrame, subset: list[str], c_value: float) -> np.ndarray:
        X_meta = meta_df[subset].to_numpy(dtype=np.float32)
        y_meta = meta_df["y_true"].to_numpy(dtype=np.int8)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_prob_1 = np.zeros(len(meta_df), dtype=np.float32)
        for fit_idx, eval_idx in cv.split(X_meta, y_meta):
            model = LogisticRegression(C=float(c_value), max_iter=2000)
            model.fit(X_meta[fit_idx], y_meta[fit_idx])
            oof_prob_1[eval_idx] = model.predict_proba(X_meta[eval_idx])[:, 1].astype(np.float32)
        return oof_prob_1


    def evaluate_logreg_candidate(meta_df: pd.DataFrame, subset: list[str], c_value: float, threshold_grid: np.ndarray) -> dict:
        prob_1 = generate_logreg_oof_probabilities(meta_df, subset, c_value)
        threshold, accuracy = find_best_threshold(meta_df["y_true"].to_numpy(dtype=np.int8), prob_1, threshold_grid)
        return {
            "meta_family": "logreg",
            "base_models": subset,
            "weights": None,
            "C": float(c_value),
            "best_threshold": threshold,
            "meta_oof_accuracy": accuracy,
        }
    """
).strip()


RUNTIME = dedent(
    """
    RUN_DISCOVERY = True
    RUN_SEARCH = True
    TRAIN_FINAL_MODEL = True

    MAX_SUBSET_SIZE = 3
    WEIGHT_STEP_SIZE_2 = 0.025
    WEIGHT_STEP_SIZE_3 = 0.05
    THRESHOLD_GRID = np.round(np.arange(0.35, 0.651, 0.01), 3)
    LOGREG_C_VALUES = [0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0]

    print(
        {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "max_subset_size": MAX_SUBSET_SIZE,
            "weight_step_size_2": WEIGHT_STEP_SIZE_2,
            "weight_step_size_3": WEIGHT_STEP_SIZE_3,
            "threshold_count": int(len(THRESHOLD_GRID)),
            "logreg_C_values": LOGREG_C_VALUES,
        }
    )
    """
).strip()


DISCOVERY = dedent(
    """
    records = discover_probability_artifacts() if RUN_DISCOVERY else []
    print("Discovered model artifacts:", [record["model_name"] for record in records])

    meta_oof_df, meta_test_df = align_artifacts(records)
    model_names = [column for column in meta_oof_df.columns if column not in {"id", "y_true"}]

    summary_df = base_model_summary_table(records, meta_oof_df)
    disagreement_df = pairwise_disagreement_table(meta_oof_df, model_names)
    correlation_df = meta_oof_df[model_names].corr()

    summary_path = CHECKPOINT_DIR / "base_model_summary.csv"
    disagreement_path = CHECKPOINT_DIR / "pairwise_disagreement.csv"
    correlation_path = CHECKPOINT_DIR / "probability_correlation.csv"

    save_dataframe_atomic(summary_df, summary_path)
    save_dataframe_atomic(disagreement_df, disagreement_path)
    save_dataframe_atomic(correlation_df.reset_index().rename(columns={"index": "model_name"}), correlation_path)

    display(summary_df)
    if not disagreement_df.empty:
        display(disagreement_df)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Correlation of base-model OOF probabilities")
    save_current_figure("oof_probability_correlation.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x="model_name", y="oof_accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.title("OOF accuracy by discovered base model")
    save_current_figure("base_model_oof_accuracy.png")

    write_json_atomic(
        CHECKPOINT_DIR / "manifest.json",
        {
            "last_checkpoint_stage": "artifact_discovery_complete",
            "discovered_models": model_names,
            "n_models": len(model_names),
            "meta_oof_rows": int(len(meta_oof_df)),
            "meta_test_rows": int(len(meta_test_df)),
        },
    )
    """
).strip()


SEARCH = dedent(
    """
    search_results_path = CHECKPOINT_DIR / "final_stacking_search_results.csv"
    search_df = read_dataframe(search_results_path)
    completed_signatures = set(search_df["signature"]) if not search_df.empty else set()

    candidate_specs = []
    next_index = 0
    subset_limit = min(MAX_SUBSET_SIZE, len(model_names))

    for subset_size in range(2, subset_limit + 1):
        for subset in itertools.combinations(model_names, subset_size):
            subset = list(subset)
            step = WEIGHT_STEP_SIZE_2 if subset_size == 2 else WEIGHT_STEP_SIZE_3
            for weights in generate_weight_vectors(subset_size, step):
                signature = json.dumps(
                    {
                        "meta_family": "weighted_average",
                        "base_models": subset,
                        "weights": weights,
                    },
                    sort_keys=True,
                )
                candidate_specs.append(
                    {
                        "candidate_id": f"blend_{next_index:04d}",
                        "signature": signature,
                        "meta_family": "weighted_average",
                        "base_models": subset,
                        "weights": weights,
                        "C": None,
                    }
                )
                next_index += 1

            for c_value in LOGREG_C_VALUES:
                signature = json.dumps(
                    {
                        "meta_family": "logreg",
                        "base_models": subset,
                        "C": float(c_value),
                    },
                    sort_keys=True,
                )
                candidate_specs.append(
                    {
                        "candidate_id": f"blend_{next_index:04d}",
                        "signature": signature,
                        "meta_family": "logreg",
                        "base_models": subset,
                        "weights": None,
                        "C": float(c_value),
                    }
                )
                next_index += 1

    print("Total stacking candidates:", len(candidate_specs))
    print("Already completed:", len(completed_signatures))

    if RUN_SEARCH:
        pending = [candidate for candidate in candidate_specs if candidate["signature"] not in completed_signatures]
        for candidate in pending:
            if candidate["meta_family"] == "weighted_average":
                result = evaluate_weighted_candidate(meta_oof_df, candidate["base_models"], candidate["weights"], THRESHOLD_GRID)
            else:
                result = evaluate_logreg_candidate(meta_oof_df, candidate["base_models"], candidate["C"], THRESHOLD_GRID)

            row = {
                "candidate_id": candidate["candidate_id"],
                "signature": candidate["signature"],
                "meta_family": result["meta_family"],
                "base_models_json": json.dumps(result["base_models"]),
                "weights_json": json.dumps(result["weights"]) if result["weights"] is not None else "",
                "C": result["C"],
                "best_threshold": result["best_threshold"],
                "meta_oof_accuracy": result["meta_oof_accuracy"],
            }
            search_df = pd.concat([search_df, pd.DataFrame([row])], ignore_index=True) if not search_df.empty else pd.DataFrame([row])
            search_df = search_df.sort_values(["meta_oof_accuracy"], ascending=[False]).reset_index(drop=True)
            save_dataframe_atomic(search_df, search_results_path)

    search_df = read_dataframe(search_results_path)
    display(search_df.head(20))

    write_json_atomic(
        CHECKPOINT_DIR / "manifest.json",
        {
            "last_checkpoint_stage": "stacking_search_complete",
            "discovered_models": model_names,
            "completed_candidates": int(len(search_df)),
            "best_meta_oof_accuracy": None if search_df.empty else float(search_df["meta_oof_accuracy"].max()),
        },
    )
    """
).strip()


PLOTS = dedent(
    """
    if not search_df.empty:
        plt.figure(figsize=(12, 6))
        plot_df = search_df.head(20).copy()
        sns.barplot(data=plot_df, x="candidate_id", y="meta_oof_accuracy", hue="meta_family")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top stacking candidates")
        save_current_figure("top_stacking_candidates.png")
    """
).strip()


FINAL = dedent(
    """
    if search_df.empty:
        raise RuntimeError("No stacking search results are available.")

    best_row = search_df.iloc[0].to_dict()
    best_base_models = json.loads(best_row["base_models_json"])
    best_threshold = float(best_row["best_threshold"])
    best_meta_family = best_row["meta_family"]

    if best_meta_family == "weighted_average":
        best_weights = json.loads(best_row["weights_json"])
        meta_oof_prob_1 = weighted_probabilities(meta_oof_df, best_base_models, best_weights).astype(np.float32)
        meta_test_prob_1 = weighted_probabilities(meta_test_df, best_base_models, best_weights).astype(np.float32)
        final_meta_description = {
            "meta_family": best_meta_family,
            "weights": best_weights,
            "C": None,
        }
    else:
        c_value = float(best_row["C"])
        meta_oof_prob_1 = generate_logreg_oof_probabilities(meta_oof_df, best_base_models, c_value).astype(np.float32)
        full_model = LogisticRegression(C=c_value, max_iter=2000)
        full_model.fit(meta_oof_df[best_base_models].to_numpy(dtype=np.float32), meta_oof_df["y_true"].to_numpy(dtype=np.int8))
        meta_test_prob_1 = full_model.predict_proba(meta_test_df[best_base_models].to_numpy(dtype=np.float32))[:, 1].astype(np.float32)
        final_meta_description = {
            "meta_family": best_meta_family,
            "weights": None,
            "C": c_value,
        }

    y_true = meta_oof_df["y_true"].to_numpy(dtype=np.int8)
    meta_oof_pred = (meta_oof_prob_1 >= best_threshold).astype(int)
    meta_test_pred = (meta_test_prob_1 >= best_threshold).astype(int)

    meta_oof_accuracy = float(accuracy_score(y_true, meta_oof_pred))
    cm = confusion_matrix(y_true, meta_oof_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Final stacking OOF confusion matrix - accuracy={meta_oof_accuracy:.4f}")
    save_current_figure("final_stacking_confusion_matrix.png")

    meta_oof_output = pd.DataFrame(
        {
            "id": meta_oof_df["id"].to_numpy(),
            "y_true": y_true,
            "prob_1": meta_oof_prob_1,
            "pred": meta_oof_pred.astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    meta_test_output = pd.DataFrame(
        {
            "id": meta_test_df["id"].to_numpy(),
            "prob_1": meta_test_prob_1,
            "pred": meta_test_pred.astype(int),
            "source_model": NOTEBOOK_SLUG,
        }
    )
    submission_df = pd.DataFrame(
        {
            "id": meta_test_df["id"].to_numpy(),
            "class": meta_test_pred.astype(int),
        }
    )

    meta_oof_path = PERSIST_ROOT / "meta_oof_probabilities.csv"
    meta_test_path = PERSIST_ROOT / "meta_test_probabilities.csv"
    submission_path = SUBMISSION_DIR / "challenge_12_final_stacking_colab_submission.csv"
    summary_path = PERSIST_ROOT / "summary.json"

    save_dataframe_atomic(meta_oof_output, meta_oof_path)
    save_dataframe_atomic(meta_test_output, meta_test_path)
    submission_df.to_csv(submission_path, index=False)

    summary_payload = {
        "model_name": "Final stacking model",
        "model_key": "final_stacking",
        "notebook_slug": NOTEBOOK_SLUG,
        "strategy": "focused_weighted_and_logreg_stacking",
        "discovered_base_models": model_names,
        "best_base_models": best_base_models,
        "best_threshold": best_threshold,
        "best_meta_oof_accuracy": meta_oof_accuracy,
        "best_candidate": final_meta_description,
        "meta_oof_path": str(meta_oof_path),
        "meta_test_path": str(meta_test_path),
        "submission_path": str(submission_path),
        "workspace_root": str(WORKSPACE_ROOT),
        "persist_root": str(PERSIST_ROOT),
    }
    write_json_atomic(summary_path, summary_payload)
    write_json_atomic(
        CHECKPOINT_DIR / "manifest.json",
        {
            "last_checkpoint_stage": "final_model_complete",
            "best_base_models": best_base_models,
            "best_threshold": best_threshold,
            "best_meta_oof_accuracy": meta_oof_accuracy,
        },
    )

    print("Best candidate:", json.dumps(summary_payload, indent=2))
    print("Submission path:", submission_path)
    """
).strip()


EXPORT = dedent(
    """
    DOWNLOAD_BUNDLE_NOW = False

    bundle_path = create_resume_bundle()
    print("Resume bundle saved to:", bundle_path)

    if DOWNLOAD_BUNDLE_NOW and IN_COLAB:
        files.download(str(bundle_path))
    """
).strip()


OUTRO = dedent(
    """
    ## Notes

    Flujo recomendado para tu caso actual:

    1. empaquetar localmente `knn_cleaning_ultra` y `signal_features_ultra`
    2. subir esos ZIPs a esta notebook
    3. correr el stacking final
    4. si luego completas `svm_preprocessing`, subir su ZIP y repetir

    Esta notebook no necesita los CSV originales del challenge.
    """
).strip()


README = dedent(
    """
    # Final Stacking for Colab

    Esta carpeta contiene la variante final de stacking para Google Colab.

    A diferencia de la notebook generica de stacking, esta version esta enfocada en tu situacion real:

    - `signal_features` como base learner fuerte
    - `knn_cleaning` como modelo de diversidad
    - `svm_preprocessing` como extra opcional

    ## Contenido

    - `Challenge_12_Final_Stacking_Colab.ipynb`
    - `package_results_pre_stack.py`
    - `requirements.txt`
    - `workspace_template/README.md`

    ## Flujo recomendado

    1. Desde tu maquina local, empaqueta tus corridas base:

       ```bash
       python3 challenge/colab_final_stacking/package_results_pre_stack.py
       ```

    2. Eso generara ZIPs dentro de `challenge/colab_final_stacking/upload_bundles/`.

    3. Sube `Challenge_12_Final_Stacking_Colab.ipynb` a Google Colab.

    4. En la celda `Optional: upload one or more model-result ZIP bundles` cambia:

       ```python
       UPLOAD_RESULT_BUNDLES = True
       ```

    5. Sube al menos dos ZIPs:
       - `knn_cleaning_ultra_for_final_stacking.zip`
       - `signal_features_ultra_for_final_stacking.zip`

    6. Ejecuta el resto de la notebook.

    7. Si despues completas SVM, genera su ZIP y vuelve a correr el stacking incluyendo ese bundle.

    ## Que busca internamente

    - `weighted average` con busqueda de pesos
    - busqueda de `threshold`
    - `LogisticRegression` como meta-modelo

    ## Artefactos finales

    - `output/challenge_12_final_stacking_colab/summary.json`
    - `output/challenge_12_final_stacking_colab/meta_oof_probabilities.csv`
    - `output/challenge_12_final_stacking_colab/meta_test_probabilities.csv`
    - `submissions/challenge_12_final_stacking_colab_submission.csv`
    """
).strip() + "\n"


REQUIREMENTS = dedent(
    """
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    """
).strip() + "\n"


WORKSPACE_TEMPLATE = dedent(
    """
    Esta carpeta es solo una referencia de la estructura esperada del workspace de la notebook final de stacking.

    Flujo normal:

    1. Subir la notebook a Colab.
    2. Subir uno o mas ZIPs con artefactos de modelos base.
    3. Ejecutar la busqueda de stacking.
    4. Descargar la submission final o el bundle de reanudacion.
    """
).strip() + "\n"


PACKAGE_RESULTS_SCRIPT = dedent(
    """
    from __future__ import annotations

    import zipfile
    from pathlib import Path


    THIS_DIR = Path(__file__).resolve().parent
    CHALLENGE_ROOT = THIS_DIR.parent
    RESULTS_ROOT = CHALLENGE_ROOT / "results-pre-stack"
    OUTPUT_DIR = THIS_DIR / "upload_bundles"


    def should_include_run(run_dir: Path) -> bool:
        output_dir = run_dir / "output"
        return any(output_dir.rglob("oof_probabilities.csv")) and any(output_dir.rglob("test_probabilities.csv"))


    def build_bundle(run_dir: Path) -> Path:
        bundle_path = OUTPUT_DIR / f"{run_dir.name}_for_final_stacking.zip"
        if bundle_path.exists():
            bundle_path.unlink()

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for top_name in ["output", "submissions"]:
                top_path = run_dir / top_name
                if not top_path.exists():
                    continue
                for nested in top_path.rglob("*"):
                    if nested.is_dir():
                        continue
                    relative_path = nested.relative_to(run_dir)
                    zip_file.write(nested, arcname=str(relative_path))

        return bundle_path


    def main() -> None:
        if not RESULTS_ROOT.exists():
            raise FileNotFoundError(f"Results root not found: {RESULTS_ROOT}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        created = []
        for run_dir in sorted(RESULTS_ROOT.iterdir()):
            if not run_dir.is_dir():
                continue
            if not should_include_run(run_dir):
                print(f"Skipping {run_dir.name}: no complete OOF/test probability artifacts found.")
                continue
            bundle_path = build_bundle(run_dir)
            created.append(bundle_path)
            print(f"Created {bundle_path}")

        if not created:
            print("No upload-ready bundles were created.")
        else:
            print("\\nUpload these bundles to the final Colab notebook:")
            for bundle_path in created:
                print("-", bundle_path)


    if __name__ == "__main__":
        main()
    """
).strip() + "\n"


def md_cell(source: str):
    return nbf.v4.new_markdown_cell(source)


def code_cell(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md_cell(INTRO),
        md_cell("## 0. Imports and global constants"),
        code_cell(IMPORTS),
        md_cell("## 1. Create the stacking workspace"),
        code_cell(SETUP),
        md_cell("## 2. Inspect the workspace"),
        code_cell(STATUS),
        md_cell("## 3. Optional: upload one or more model-result ZIP bundles"),
        code_cell(UPLOAD_BUNDLES),
        md_cell("## 4. Helpers"),
        code_cell(HELPERS),
        md_cell("## 5. Runtime configuration"),
        code_cell(RUNTIME),
        md_cell("## 6. Discover uploaded artifacts and compute diagnostics"),
        code_cell(DISCOVERY),
        md_cell("## 7. Search final blending and meta-model candidates"),
        code_cell(SEARCH),
        md_cell("## 8. Diagnostic plots"),
        code_cell(PLOTS),
        md_cell("## 9. Train the final stacking solution and create the submission"),
        code_cell(FINAL),
        md_cell("## 10. Optional: export and download a resume bundle"),
        code_cell(EXPORT),
        md_cell(OUTRO),
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
    WORKSPACE_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    README_PATH.write_text(README)
    REQUIREMENTS_PATH.write_text(REQUIREMENTS)
    WORKSPACE_TEMPLATE_PATH.write_text(WORKSPACE_TEMPLATE)
    (TARGET_DIR / "package_results_pre_stack.py").write_text(PACKAGE_RESULTS_SCRIPT)

    print(f"Created {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
