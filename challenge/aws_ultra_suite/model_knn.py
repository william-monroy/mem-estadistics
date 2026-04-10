from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import (
    AwsUltraConfig,
    append_rows,
    build_model_context,
    checkpoint_housekeeping,
    cpu_runtime_snapshot,
    evaluate_predictions,
    load_base_data,
    maybe_sync_model_outputs_to_s3,
    normalize_value,
    read_dataframe,
    render_validation_confusion_matrix,
    save_current_figure,
    save_dataframe_atomic,
    stage_log,
    update_manifest,
    write_json_atomic,
    write_model_summary,
)


MODEL_KEY = "knn"
MODEL_NAME = "K-nearest neighbors"
SLUG = "challenge_01_knn_aws_ultra"


SEARCH_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "stage1_n_iter": 96,
        "stage1_pool_fraction": 0.70,
        "stage1_eval_size": 0.25,
        "stage2_top_k": 20,
        "stage2_cv": 3,
        "stage3_seed_top_k": 3,
        "stage3_cv": 5,
        "stage4_top_k": 5,
        "stage4_repeats": 5,
        "stage4_eval_size": 0.25,
    },
    "aws_fast": {
        "stage1_n_iter": 160,
        "stage1_pool_fraction": 0.90,
        "stage1_eval_size": 0.25,
        "stage2_top_k": 28,
        "stage2_cv": 3,
        "stage3_seed_top_k": 4,
        "stage3_cv": 5,
        "stage4_top_k": 6,
        "stage4_repeats": 8,
        "stage4_eval_size": 0.25,
    },
    "aws_max": {
        "stage1_n_iter": 220,
        "stage1_pool_fraction": 1.00,
        "stage1_eval_size": 0.25,
        "stage2_top_k": 36,
        "stage2_cv": 3,
        "stage3_seed_top_k": 5,
        "stage3_cv": 5,
        "stage4_top_k": 8,
        "stage4_repeats": 10,
        "stage4_eval_size": 0.25,
    },
}


def normalize_candidate(params: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: normalize_value(value) for key, value in params.items()}
    normalized["pca__n_components"] = int(normalized["pca__n_components"])
    normalized["model__n_neighbors"] = int(normalized["model__n_neighbors"])
    normalized["model__weights"] = str(normalized["model__weights"])
    normalized["model__metric"] = str(normalized["model__metric"])
    return normalized


def candidate_signature(params: dict[str, Any]) -> str:
    return json.dumps(normalize_candidate(params), sort_keys=True)


def candidate_record(candidate_id: str, params: dict[str, Any], source: str) -> dict[str, Any]:
    params = normalize_candidate(params)
    return {
        "candidate_id": candidate_id,
        "source": source,
        "params": params,
        "signature": candidate_signature(params),
    }


def candidate_to_row(candidate: dict[str, Any]) -> dict[str, Any]:
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


def build_knn(params: dict[str, Any], n_jobs: int) -> KNeighborsClassifier:
    params = normalize_candidate(params)
    return KNeighborsClassifier(
        n_neighbors=int(params["model__n_neighbors"]),
        weights=params["model__weights"],
        metric=params["model__metric"],
        algorithm="brute",
        n_jobs=n_jobs,
    )


def fit_projection(X_train_input: np.ndarray, X_eval_input: np.ndarray, max_components: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_input)
    X_eval_scaled = scaler.transform(X_eval_input)
    safe_components = min(int(max_components), X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=safe_components, random_state=random_state)
    X_train_proj = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_eval_proj = pca.transform(X_eval_scaled).astype(np.float32)
    return X_train_proj, X_eval_proj


def evaluate_candidate(
    X_train_proj: np.ndarray,
    X_eval_proj: np.ndarray,
    y_train_input: np.ndarray,
    y_eval_input: np.ndarray,
    candidate: dict[str, Any],
    *,
    n_jobs: int,
) -> tuple[float, float]:
    params = candidate["params"]
    n_components = int(params["pca__n_components"])
    model = build_knn(params, n_jobs=n_jobs)
    start_time = time.time()
    model.fit(X_train_proj[:, :n_components], y_train_input)
    predictions = model.predict(X_eval_proj[:, :n_components])
    elapsed = time.time() - start_time
    accuracy = (predictions == y_eval_input).mean()
    return float(accuracy), round(elapsed, 3)


def fit_knn_bundle(X_fit: np.ndarray, y_fit: np.ndarray, params: dict[str, Any], *, random_state: int, n_jobs: int) -> dict[str, Any]:
    params = normalize_candidate(params)
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    pca = PCA(n_components=int(params["pca__n_components"]), random_state=random_state)
    X_fit_proj = pca.fit_transform(X_fit_scaled).astype(np.float32)
    model = build_knn(params, n_jobs=n_jobs)
    model.fit(X_fit_proj, y_fit)
    return {"scaler": scaler, "pca": pca, "model": model, "params": params}


def predict_knn_bundle(bundle: dict[str, Any], X_input: np.ndarray) -> np.ndarray:
    X_scaled = bundle["scaler"].transform(X_input)
    X_proj = bundle["pca"].transform(X_scaled).astype(np.float32)
    return bundle["model"].predict(X_proj)


def run(config: AwsUltraConfig) -> Path:
    profile = SEARCH_PRESETS.get(config.profile, SEARCH_PRESETS["aws_fast"])
    context = build_model_context(config, MODEL_KEY, SLUG, MODEL_NAME)
    knn_n_jobs = config.resolved_knn_n_jobs()

    stage_log(f"{MODEL_NAME}: starting with profile={config.profile}")
    stage_log(f"{MODEL_NAME}: runtime snapshot -> {cpu_runtime_snapshot(config)}")

    base = load_base_data(config)
    train_df = base["train_df"]
    test_df = base["test_df"]
    X_full = base["X_full"]
    y_full = base["y_full"]
    X_test_full = base["X_test_full"]
    X_train = base["X_train"]
    X_valid = base["X_valid"]
    y_train = base["y_train"]
    y_valid = base["y_valid"]

    if profile["stage1_pool_fraction"] < 1.0:
        X_stage1_pool, _, y_stage1_pool, _ = train_test_split(
            X_train,
            y_train,
            train_size=profile["stage1_pool_fraction"],
            stratify=y_train,
            random_state=config.random_state,
        )
    else:
        X_stage1_pool, y_stage1_pool = X_train, y_train

    X_stage1_fit, X_stage1_eval, y_stage1_fit, y_stage1_eval = train_test_split(
        X_stage1_pool,
        y_stage1_pool,
        test_size=profile["stage1_eval_size"],
        stratify=y_stage1_pool,
        random_state=config.random_state,
    )

    stage_log(f"{MODEL_NAME}: train={X_train.shape}, valid={X_valid.shape}, stage1_fit={X_stage1_fit.shape}, stage1_eval={X_stage1_eval.shape}")

    stage1_candidates_path = context.checkpoint_dir / "stage1_candidates.json"
    stage1_results_path = context.checkpoint_dir / "stage1_holdout_results.csv"
    stage2_fold_path = context.checkpoint_dir / "stage2_cv_fold_results.csv"
    stage2_summary_path = context.checkpoint_dir / "stage2_cv_summary.csv"
    stage3_candidates_path = context.checkpoint_dir / "stage3_local_candidates.json"
    stage3_fold_path = context.checkpoint_dir / "stage3_local_cv_fold_results.csv"
    stage3_summary_path = context.checkpoint_dir / "stage3_local_cv_summary.csv"
    stage4_split_path = context.checkpoint_dir / "stage4_stability_split_results.csv"
    stage4_summary_path = context.checkpoint_dir / "stage4_stability_summary.csv"

    stage1_space = {
        "pca__n_components": [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 112, 128],
        "model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["manhattan", "euclidean", "chebyshev"],
    }

    if stage1_candidates_path.exists():
        stage1_candidates = json.loads(stage1_candidates_path.read_text())
    else:
        sampler = ParameterSampler(stage1_space, n_iter=profile["stage1_n_iter"], random_state=config.random_state)
        seen: set[str] = set()
        stage1_candidates = []
        for sampled_params in sampler:
            normalized = normalize_candidate(sampled_params)
            signature = candidate_signature(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            stage1_candidates.append(candidate_record(f"stage1_{len(stage1_candidates):03d}", normalized, "stage1_random_holdout"))
        write_json_atomic(stage1_candidates_path, stage1_candidates)

    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()

    stage_log(f"{MODEL_NAME}: stage1 candidates={len(stage1_candidates)}, completed={len(completed_stage1)}")
    if len(completed_stage1) < len(stage1_candidates):
        max_components_stage1 = max(candidate["params"]["pca__n_components"] for candidate in stage1_candidates)
        X_stage1_fit_proj, X_stage1_eval_proj = fit_projection(
            X_stage1_fit,
            X_stage1_eval,
            max_components_stage1,
            config.random_state,
        )
        for candidate in [item for item in stage1_candidates if item["candidate_id"] not in completed_stage1]:
            accuracy, elapsed = evaluate_candidate(
                X_stage1_fit_proj,
                X_stage1_eval_proj,
                y_stage1_fit,
                y_stage1_eval,
                candidate,
                n_jobs=knn_n_jobs,
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
                context,
                {
                    "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                    "stage1_total_candidates": len(stage1_candidates),
                    "stage1_best_holdout_accuracy": float(stage1_results["holdout_accuracy"].max()),
                },
            )

    stage1_results = read_dataframe(stage1_results_path)
    stage1_results = stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True]).reset_index(drop=True)
    checkpoint_housekeeping(context, "stage1_complete", refresh_bundle=True)

    stage2_shortlist = (
        stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(profile["stage2_top_k"])
        .copy()
    )
    stage2_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage2_shortlist_from_stage1")
        for row in stage2_shortlist.to_dict(orient="records")
    ]
    stage2_cv = StratifiedKFold(n_splits=profile["stage2_cv"], shuffle=True, random_state=config.random_state)
    stage2_splits = list(stage2_cv.split(X_train, y_train))

    def refresh_stage2_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage2_fold_path)
        if fold_df.empty:
            return pd.DataFrame()
        summary_rows = []
        for candidate in stage2_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < profile["stage2_cv"]:
                continue
            summary_rows.append(
                {
                    **candidate_to_row(candidate),
                    "stage": "stage2_cv",
                    "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                    "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                    "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                    "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage2_summary_path)
        return summary_df

    stage2_summary = refresh_stage2_summary()
    fold_df = read_dataframe(stage2_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()
    stage_log(f"{MODEL_NAME}: stage2 candidates={len(stage2_candidates)}, completed_pairs={len(completed_pairs)}")

    max_components_stage2 = max(candidate["params"]["pca__n_components"] for candidate in stage2_candidates)
    for fold_idx, (train_idx, test_idx) in enumerate(stage2_splits):
        pending_candidates = [candidate for candidate in stage2_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
        if not pending_candidates:
            continue
        X_fold_train_proj, X_fold_test_proj = fit_projection(
            X_train[train_idx],
            X_train[test_idx],
            max_components_stage2,
            config.random_state + fold_idx,
        )
        for candidate in pending_candidates:
            accuracy, elapsed = evaluate_candidate(
                X_fold_train_proj,
                X_fold_test_proj,
                y_train[train_idx],
                y_train[test_idx],
                candidate,
                n_jobs=knn_n_jobs,
            )
            append_rows(
                stage2_fold_path,
                [
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage2_cv",
                        "fold_idx": fold_idx,
                        "fold_accuracy": accuracy,
                        "fit_seconds": elapsed,
                    }
                ],
                sort_cols=["candidate_id", "fold_idx"],
            )
            completed_pairs.add((candidate["candidate_id"], fold_idx))
        stage2_summary = refresh_stage2_summary()
        if not stage2_summary.empty:
            update_manifest(
                context,
                {
                    "stage2_completed_candidates": int(stage2_summary["candidate_id"].nunique()),
                    "stage2_total_candidates": len(stage2_candidates),
                    "stage2_best_cv_accuracy": float(stage2_summary["cv_mean_accuracy"].max()),
                },
            )
        gc.collect()

    stage2_summary = read_dataframe(stage2_summary_path)
    checkpoint_housekeeping(context, "stage2_complete", refresh_bundle=True)

    stage3_seed_rows = (
        stage2_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(profile["stage3_seed_top_k"])
        .copy()
    )

    if stage3_candidates_path.exists():
        stage3_candidates = json.loads(stage3_candidates_path.read_text())
    else:
        candidates = []
        seen: set[str] = set()
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
                        best_components - 12,
                        best_components - 8,
                        best_components - 4,
                        best_components - 2,
                        best_components,
                        best_components + 2,
                        best_components + 4,
                        best_components + 8,
                        best_components + 12,
                    ]
                    if 8 <= value <= X_train.shape[1]
                }
            )
            neighbor_values = sorted(
                {
                    value
                    for value in [
                        best_k - 4,
                        best_k - 3,
                        best_k - 2,
                        best_k - 1,
                        best_k,
                        best_k + 1,
                        best_k + 2,
                        best_k + 3,
                        best_k + 4,
                        best_k + 6,
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
                            candidates.append(candidate_record(f"stage3_{next_index:03d}", params, f"stage3_seed_{seed_rank}"))
                            next_index += 1
        stage3_candidates = candidates
        write_json_atomic(stage3_candidates_path, stage3_candidates)

    stage3_cv = StratifiedKFold(n_splits=profile["stage3_cv"], shuffle=True, random_state=config.random_state)
    stage3_splits = list(stage3_cv.split(X_train, y_train))

    def refresh_stage3_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage3_fold_path)
        if fold_df.empty:
            return pd.DataFrame()
        summary_rows = []
        for candidate in stage3_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < profile["stage3_cv"]:
                continue
            summary_rows.append(
                {
                    **candidate_to_row(candidate),
                    "stage": "stage3_local_cv",
                    "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                    "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                    "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                    "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage3_summary_path)
        return summary_df

    stage3_summary = refresh_stage3_summary()
    fold_df = read_dataframe(stage3_fold_path)
    completed_pairs = set(zip(fold_df["candidate_id"].astype(str), fold_df["fold_idx"].astype(int))) if not fold_df.empty else set()
    stage_log(f"{MODEL_NAME}: stage3 candidates={len(stage3_candidates)}, completed_pairs={len(completed_pairs)}")

    max_components_stage3 = max(candidate["params"]["pca__n_components"] for candidate in stage3_candidates)
    for fold_idx, (train_idx, test_idx) in enumerate(stage3_splits):
        pending_candidates = [candidate for candidate in stage3_candidates if (candidate["candidate_id"], fold_idx) not in completed_pairs]
        if not pending_candidates:
            continue
        X_fold_train_proj, X_fold_test_proj = fit_projection(
            X_train[train_idx],
            X_train[test_idx],
            max_components_stage3,
            config.random_state + 100 + fold_idx,
        )
        for candidate in pending_candidates:
            accuracy, elapsed = evaluate_candidate(
                X_fold_train_proj,
                X_fold_test_proj,
                y_train[train_idx],
                y_train[test_idx],
                candidate,
                n_jobs=knn_n_jobs,
            )
            append_rows(
                stage3_fold_path,
                [
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage3_local_cv",
                        "fold_idx": fold_idx,
                        "fold_accuracy": accuracy,
                        "fit_seconds": elapsed,
                    }
                ],
                sort_cols=["candidate_id", "fold_idx"],
            )
            completed_pairs.add((candidate["candidate_id"], fold_idx))
        stage3_summary = refresh_stage3_summary()
        if not stage3_summary.empty:
            update_manifest(
                context,
                {
                    "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                    "stage3_total_candidates": len(stage3_candidates),
                    "stage3_best_cv_accuracy": float(stage3_summary["cv_mean_accuracy"].max()),
                },
            )
        gc.collect()

    stage3_summary = read_dataframe(stage3_summary_path)
    checkpoint_housekeeping(context, "stage3_complete", refresh_bundle=True)

    stage4_seed_rows = (
        stage3_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(profile["stage4_top_k"])
        .copy()
    )
    stage4_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage4_top_from_stage3")
        for row in stage4_seed_rows.to_dict(orient="records")
    ]

    def refresh_stage4_summary() -> pd.DataFrame:
        split_df = read_dataframe(stage4_split_path)
        if split_df.empty:
            return pd.DataFrame()
        summary_rows = []
        for candidate in stage4_candidates:
            candidate_split_df = split_df[split_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_split_df) < profile["stage4_repeats"]:
                continue
            summary_rows.append(
                {
                    **candidate_to_row(candidate),
                    "stage": "stage4_stability",
                    "mean_accuracy": float(candidate_split_df["split_accuracy"].mean()),
                    "std_accuracy": float(candidate_split_df["split_accuracy"].std(ddof=0)),
                    "total_fit_seconds": float(candidate_split_df["fit_seconds"].sum()),
                    "completed_splits": int(candidate_split_df["split_idx"].nunique()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["mean_accuracy", "std_accuracy", "total_fit_seconds"], ascending=[False, True, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage4_summary_path)
        return summary_df

    stage4_summary = refresh_stage4_summary()
    split_df = read_dataframe(stage4_split_path)
    completed_pairs = set(zip(split_df["candidate_id"].astype(str), split_df["split_idx"].astype(int))) if not split_df.empty else set()
    stage_log(f"{MODEL_NAME}: stage4 candidates={len(stage4_candidates)}, completed_pairs={len(completed_pairs)}")

    max_components_stage4 = max(candidate["params"]["pca__n_components"] for candidate in stage4_candidates)
    for split_idx in range(profile["stage4_repeats"]):
        pending_candidates = [candidate for candidate in stage4_candidates if (candidate["candidate_id"], split_idx) not in completed_pairs]
        if not pending_candidates:
            continue
        X_split_train, X_split_eval, y_split_train, y_split_eval = train_test_split(
            X_train,
            y_train,
            test_size=profile["stage4_eval_size"],
            stratify=y_train,
            random_state=config.random_state + 500 + split_idx,
        )
        X_split_train_proj, X_split_eval_proj = fit_projection(
            X_split_train,
            X_split_eval,
            max_components_stage4,
            config.random_state + 500 + split_idx,
        )
        for candidate in pending_candidates:
            accuracy, elapsed = evaluate_candidate(
                X_split_train_proj,
                X_split_eval_proj,
                y_split_train,
                y_split_eval,
                candidate,
                n_jobs=knn_n_jobs,
            )
            append_rows(
                stage4_split_path,
                [
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage4_stability",
                        "split_idx": split_idx,
                        "split_accuracy": accuracy,
                        "fit_seconds": elapsed,
                    }
                ],
                sort_cols=["candidate_id", "split_idx"],
            )
            completed_pairs.add((candidate["candidate_id"], split_idx))
        stage4_summary = refresh_stage4_summary()
        if not stage4_summary.empty:
            update_manifest(
                context,
                {
                    "stage4_completed_candidates": int(stage4_summary["candidate_id"].nunique()),
                    "stage4_total_candidates": len(stage4_candidates),
                    "stage4_best_mean_accuracy": float(stage4_summary["mean_accuracy"].max()),
                },
            )
        gc.collect()

    stage4_summary = read_dataframe(stage4_summary_path)
    checkpoint_housekeeping(context, "stage4_complete", refresh_bundle=True)

    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        top_stage1 = stage1_results.head(12).copy()
        top_stage1["label"] = top_stage1["candidate_id"] + " | c=" + top_stage1["pca__n_components"].astype(int).astype(str) + " | k=" + top_stage1["model__n_neighbors"].astype(int).astype(str)
        sns.barplot(data=top_stage1, x="holdout_accuracy", y="label", palette="crest")
        plt.title("Top Stage 1 KNN candidates by holdout accuracy")
        plt.xlabel("Holdout accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage1_top_candidates.png")

    if not stage2_summary.empty:
        stage2_plot = stage2_summary.copy()
        best_metric = stage2_plot.iloc[0]["model__metric"]
        best_weight = stage2_plot.iloc[0]["model__weights"]
        heatmap_data = (
            stage2_plot[
                (stage2_plot["model__metric"] == best_metric)
                & (stage2_plot["model__weights"] == best_weight)
            ]
            .pivot_table(index="pca__n_components", columns="model__n_neighbors", values="cv_mean_accuracy")
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Stage 2 CV heatmap\nmetric={best_metric}, weights={best_weight}")
        plt.xlabel("Number of neighbors")
        plt.ylabel("PCA components")
        save_current_figure(context.output_dir / "stage2_cv_heatmap.png")

    if not stage3_summary.empty:
        stage3_plot = stage3_summary.copy()
        best_metric = stage3_plot.iloc[0]["model__metric"]
        best_weight = stage3_plot.iloc[0]["model__weights"]
        heatmap_data = (
            stage3_plot[
                (stage3_plot["model__metric"] == best_metric)
                & (stage3_plot["model__weights"] == best_weight)
            ]
            .pivot_table(index="pca__n_components", columns="model__n_neighbors", values="cv_mean_accuracy")
            .sort_index()
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="mako")
        plt.title(f"Stage 3 local refinement heatmap\nmetric={best_metric}, weights={best_weight}")
        plt.xlabel("Number of neighbors")
        plt.ylabel("PCA components")
        save_current_figure(context.output_dir / "stage3_local_heatmap.png")

    if not stage4_summary.empty:
        top_stage4 = stage4_summary.head(10).copy()
        top_stage4["label"] = top_stage4["candidate_id"] + " | c=" + top_stage4["pca__n_components"].astype(int).astype(str) + " | k=" + top_stage4["model__n_neighbors"].astype(int).astype(str)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_stage4, x="mean_accuracy", y="label", palette="rocket")
        plt.title("Top Stage 4 KNN candidates by stability mean accuracy")
        plt.xlabel("Mean repeated-holdout accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage4_stability_top_candidates.png")

    if not stage4_summary.empty:
        final_search_table = stage4_summary.copy()
        final_stage_name = "stage4_stability"
        best_row = final_search_table.sort_values(["mean_accuracy", "std_accuracy", "total_fit_seconds"], ascending=[False, True, True]).iloc[0]
        best_score = float(best_row["mean_accuracy"])
    elif not stage3_summary.empty:
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
        final_stage_name = "stage1_holdout"
        best_row = stage1_results.sort_values(["holdout_accuracy", "fit_seconds"], ascending=[False, True]).iloc[0]
        best_score = float(best_row["holdout_accuracy"])

    best_params = json.loads(best_row["params_json"])
    stage_log(f"{MODEL_NAME}: final stage={final_stage_name}, best params={best_params}, best score={best_score:.4f}")

    final_bundle = fit_knn_bundle(X_train, y_train, best_params, random_state=config.random_state, n_jobs=knn_n_jobs)
    valid_pred = predict_knn_bundle(final_bundle, X_valid)
    validation_report = evaluate_predictions(y_valid, valid_pred)
    render_validation_confusion_matrix(context.output_dir, y_valid, valid_pred, "Validation confusion matrix")

    full_bundle = fit_knn_bundle(X_full, y_full, best_params, random_state=config.random_state, n_jobs=knn_n_jobs)
    kaggle_pred = predict_knn_bundle(full_bundle, X_test_full)

    submission_df = pd.DataFrame({"id": test_df["id"], "class": kaggle_pred.astype(int)})
    submission_path = config.submissions_root / f"{SLUG}_submission.csv"
    submission_df.to_csv(submission_path, index=False)

    summary_path = write_model_summary(
        context,
        {
            "model_key": MODEL_KEY,
            "model_name": MODEL_NAME,
            "slug": SLUG,
            "profile": config.profile,
            "knn_n_jobs": knn_n_jobs,
            "best_stage": final_stage_name,
            "best_params": best_params,
            "best_search_stage_score": best_score,
            "validation_accuracy": float(validation_report["accuracy"]),
            "submission_path": str(submission_path),
            "workspace_root": str(config.workspace_root),
            "output_dir": str(context.output_dir),
            "train_shape": list(train_df.shape),
        },
    )
    checkpoint_housekeeping(context, "final_model_complete", refresh_bundle=True)
    maybe_sync_model_outputs_to_s3(config, context)
    stage_log(f"{MODEL_NAME}: completed -> {summary_path}")
    return summary_path
