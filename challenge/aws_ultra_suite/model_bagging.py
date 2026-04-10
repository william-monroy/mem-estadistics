from __future__ import annotations

import gc
import json
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

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


MODEL_KEY = "bagging"
MODEL_NAME = "Bagging over decision trees"
SLUG = "challenge_07_bagging_aws_ultra"


SEARCH_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "stage1_n_iter": 80,
        "stage1_train_fraction": 0.70,
        "stage1_n_estimators": 160,
        "stage2_top_k": 16,
        "stage2_cv": 3,
        "stage2_n_estimators": 320,
        "stage3_top_k": 4,
        "stage3_cv": 5,
        "stage3_tree_schedule": [320, 512, 768],
        "stage4_cv": 5,
    },
    "aws_fast": {
        "stage1_n_iter": 112,
        "stage1_train_fraction": 0.85,
        "stage1_n_estimators": 192,
        "stage2_top_k": 20,
        "stage2_cv": 3,
        "stage2_n_estimators": 384,
        "stage3_top_k": 5,
        "stage3_cv": 5,
        "stage3_tree_schedule": [384, 640, 896, 1152],
        "stage4_cv": 5,
    },
    "aws_max": {
        "stage1_n_iter": 144,
        "stage1_train_fraction": 0.90,
        "stage1_n_estimators": 256,
        "stage2_top_k": 24,
        "stage2_cv": 3,
        "stage2_n_estimators": 512,
        "stage3_top_k": 6,
        "stage3_cv": 5,
        "stage3_tree_schedule": [512, 768, 1024, 1280],
        "stage4_cv": 5,
    },
}


def get_supported_criteria() -> list[str]:
    version_parts = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    criteria = ["gini", "entropy"]
    if version_parts >= (1, 1):
        criteria.append("log_loss")
    return criteria


TREE_CRITERIA = get_supported_criteria()


def normalize_candidate(params: dict[str, Any]) -> dict[str, Any]:
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
        "base__criterion": params.get("base__criterion"),
        "base__max_depth": params.get("base__max_depth"),
        "base__min_samples_leaf": params.get("base__min_samples_leaf"),
        "base__min_samples_split": params.get("base__min_samples_split"),
        "model__max_features": params.get("model__max_features"),
        "model__max_samples": params.get("model__max_samples"),
        "model__bootstrap_features": params.get("model__bootstrap_features"),
        "params_json": json.dumps(params, sort_keys=True),
    }


def build_base_tree(params: dict[str, Any], *, random_state: int) -> DecisionTreeClassifier:
    params = normalize_candidate(params)
    return DecisionTreeClassifier(
        criterion=params["base__criterion"],
        max_depth=params["base__max_depth"],
        min_samples_leaf=params["base__min_samples_leaf"],
        min_samples_split=params["base__min_samples_split"],
        random_state=random_state,
    )


def build_bagging(
    params: dict[str, Any],
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    oob_score: bool = False,
    warm_start: bool = False,
) -> BaggingClassifier:
    params = normalize_candidate(params)
    estimator = build_base_tree(params, random_state=random_state)
    kwargs = {
        "n_estimators": int(n_estimators),
        "bootstrap": True,
        "oob_score": oob_score,
        "warm_start": warm_start,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "max_samples": params["model__max_samples"],
        "max_features": params["model__max_features"],
        "bootstrap_features": params["model__bootstrap_features"],
    }
    try:
        return BaggingClassifier(estimator=estimator, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=estimator, **kwargs)


def run(config: AwsUltraConfig) -> Path:
    profile = SEARCH_PRESETS.get(config.profile, SEARCH_PRESETS["aws_fast"])
    context = build_model_context(config, MODEL_KEY, SLUG, MODEL_NAME)
    ensemble_n_jobs = config.resolved_ensemble_n_jobs()

    stage_log(f"{MODEL_NAME}: starting with profile={config.profile}")
    stage_log(f"{MODEL_NAME}: runtime snapshot -> {cpu_runtime_snapshot(config)}")
    stage_log(f"{MODEL_NAME}: n_jobs={ensemble_n_jobs}, criteria={TREE_CRITERIA}")

    base = load_base_data(config)
    train_df = base["train_df"]
    test_df = base["test_df"]
    feature_names = base["feature_names"]
    X_full = base["X_full"]
    y_full = base["y_full"]
    X_test_full = base["X_test_full"]
    X_train = base["X_train"]
    X_valid = base["X_valid"]
    y_train = base["y_train"]
    y_valid = base["y_valid"]

    if profile["stage1_train_fraction"] < 1.0:
        X_stage1, _, y_stage1, _ = train_test_split(
            X_train,
            y_train,
            train_size=profile["stage1_train_fraction"],
            stratify=y_train,
            random_state=config.random_state,
        )
    else:
        X_stage1, y_stage1 = X_train, y_train

    stage_log(f"{MODEL_NAME}: train={X_train.shape}, valid={X_valid.shape}, stage1={X_stage1.shape}")

    stage1_candidates_path = context.checkpoint_dir / "stage1_candidates.json"
    stage1_results_path = context.checkpoint_dir / "stage1_oob_results.csv"
    stage2_fold_path = context.checkpoint_dir / "stage2_cv_fold_results.csv"
    stage2_summary_path = context.checkpoint_dir / "stage2_cv_summary.csv"
    stage3_fold_path = context.checkpoint_dir / "stage3_ensemble_growth_fold_results.csv"
    stage3_summary_path = context.checkpoint_dir / "stage3_ensemble_growth_summary.csv"
    stage4_candidates_path = context.checkpoint_dir / "stage4_local_candidates.json"
    stage4_fold_path = context.checkpoint_dir / "stage4_local_cv_fold_results.csv"
    stage4_summary_path = context.checkpoint_dir / "stage4_local_cv_summary.csv"

    stage1_space = {
        "base__criterion": TREE_CRITERIA,
        "base__max_depth": [None, 8, 12, 16, 20, 28],
        "base__min_samples_leaf": [1, 2, 4, 6, 8],
        "base__min_samples_split": [2, 4, 8, 12, 16],
        "model__max_features": [0.25, 0.35, 0.50, 0.65, 0.80, 1.0],
        "model__max_samples": [0.50, 0.65, 0.80, 1.0],
        "model__bootstrap_features": [False, True],
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
            stage1_candidates.append(candidate_record(f"stage1_{len(stage1_candidates):03d}", normalized, "stage1_random_oob"))
        write_json_atomic(stage1_candidates_path, stage1_candidates)

    stage1_results = read_dataframe(stage1_results_path)
    completed_stage1 = set(stage1_results["candidate_id"]) if not stage1_results.empty else set()
    stage_log(f"{MODEL_NAME}: stage1 candidates={len(stage1_candidates)}, completed={len(completed_stage1)}")

    for candidate in [item for item in stage1_candidates if item["candidate_id"] not in completed_stage1]:
        start_time = time.time()
        model = build_bagging(
            candidate["params"],
            n_estimators=profile["stage1_n_estimators"],
            random_state=config.random_state,
            n_jobs=ensemble_n_jobs,
            oob_score=True,
        )
        model.fit(X_stage1, y_stage1)
        elapsed = time.time() - start_time
        stage1_results = append_rows(
            stage1_results_path,
            [
                {
                    **candidate_to_row(candidate),
                    "stage": "stage1_oob",
                    "n_estimators": profile["stage1_n_estimators"],
                    "oob_accuracy": float(model.oob_score_),
                    "fit_seconds": round(elapsed, 3),
                }
            ],
            sort_cols=["oob_accuracy", "fit_seconds"],
            ascending=[False, True],
        )
        update_manifest(
            context,
            {
                "stage1_completed_candidates": int(stage1_results["candidate_id"].nunique()),
                "stage1_total_candidates": len(stage1_candidates),
                "stage1_best_oob_accuracy": float(stage1_results["oob_accuracy"].max()),
            },
        )
        del model
        gc.collect()

    stage1_results = read_dataframe(stage1_results_path).sort_values(["oob_accuracy", "fit_seconds"], ascending=[False, True]).reset_index(drop=True)
    checkpoint_housekeeping(context, "stage1_complete", refresh_bundle=True)

    stage2_shortlist = (
        stage1_results.drop_duplicates("signature")
        .sort_values(["oob_accuracy", "fit_seconds"], ascending=[False, True])
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
                    "n_estimators": profile["stage2_n_estimators"],
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
    completed_stage2 = set(stage2_summary["candidate_id"]) if not stage2_summary.empty else set()
    stage_log(f"{MODEL_NAME}: stage2 candidates={len(stage2_candidates)}, completed={len(completed_stage2)}")

    for candidate in [item for item in stage2_candidates if item["candidate_id"] not in completed_stage2]:
        fold_df = read_dataframe(stage2_fold_path)
        done_folds = set(fold_df.loc[fold_df["candidate_id"] == candidate["candidate_id"], "fold_idx"].astype(int).tolist()) if not fold_df.empty else set()
        for fold_idx, (train_idx, test_idx) in enumerate(stage2_splits):
            if fold_idx in done_folds:
                continue
            start_time = time.time()
            model = build_bagging(
                candidate["params"],
                n_estimators=profile["stage2_n_estimators"],
                random_state=config.random_state + fold_idx,
                n_jobs=ensemble_n_jobs,
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            fold_pred = model.predict(X_train[test_idx])
            fold_accuracy = float((fold_pred == y_train[test_idx]).mean())
            elapsed = time.time() - start_time
            append_rows(
                stage2_fold_path,
                [
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage2_cv",
                        "fold_idx": fold_idx,
                        "n_estimators": profile["stage2_n_estimators"],
                        "fold_accuracy": fold_accuracy,
                        "fit_seconds": round(elapsed, 3),
                    }
                ],
                sort_cols=["candidate_id", "fold_idx"],
            )
            del model
            gc.collect()
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

    stage2_summary = read_dataframe(stage2_summary_path)
    checkpoint_housekeeping(context, "stage2_complete", refresh_bundle=True)

    stage3_seed = (
        stage2_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True])
        .drop_duplicates("signature")
        .head(profile["stage3_top_k"])
        .copy()
    )
    stage3_candidates = [
        candidate_record(row["candidate_id"], json.loads(row["params_json"]), "stage3_finalists_from_stage2")
        for row in stage3_seed.to_dict(orient="records")
    ]
    stage3_cv = StratifiedKFold(n_splits=profile["stage3_cv"], shuffle=True, random_state=config.random_state)
    stage3_splits = list(stage3_cv.split(X_train, y_train))
    tree_schedule = profile["stage3_tree_schedule"]

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
    stage_log(f"{MODEL_NAME}: stage3 candidates={len(stage3_candidates)}")

    for candidate in stage3_candidates:
        fold_df = read_dataframe(stage3_fold_path)
        completed = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]] if not fold_df.empty else pd.DataFrame()
        grouped = completed.groupby("fold_idx")["n_estimators"].nunique() if not completed.empty else pd.Series(dtype=int)
        completed_folds = set(grouped[grouped == len(tree_schedule)].index.astype(int).tolist()) if not grouped.empty else set()
        for fold_idx, (train_idx, test_idx) in enumerate(stage3_splits):
            if fold_idx in completed_folds:
                continue
            model = build_bagging(
                candidate["params"],
                n_estimators=tree_schedule[0],
                random_state=config.random_state + fold_idx,
                n_jobs=ensemble_n_jobs,
                warm_start=True,
            )
            fold_rows = []
            for n_estimators in tree_schedule:
                start_time = time.time()
                model.set_params(n_estimators=int(n_estimators))
                model.fit(X_train[train_idx], y_train[train_idx])
                fold_pred = model.predict(X_train[test_idx])
                fold_accuracy = float((fold_pred == y_train[test_idx]).mean())
                elapsed = time.time() - start_time
                fold_rows.append(
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage3_ensemble_growth",
                        "fold_idx": fold_idx,
                        "n_estimators": int(n_estimators),
                        "fold_accuracy": fold_accuracy,
                        "fit_seconds": round(elapsed, 3),
                    }
                )
            append_rows(stage3_fold_path, fold_rows, sort_cols=["candidate_id", "fold_idx", "n_estimators"])
            del model
            gc.collect()
        stage3_summary = refresh_stage3_summary()
        if not stage3_summary.empty:
            update_manifest(
                context,
                {
                    "stage3_completed_candidates": int(stage3_summary["candidate_id"].nunique()),
                    "stage3_best_cv_accuracy": float(stage3_summary["cv_mean_accuracy"].max()),
                },
            )

    stage3_summary = read_dataframe(stage3_summary_path)
    checkpoint_housekeeping(context, "stage3_complete", refresh_bundle=True)

    best_stage3_row = stage3_summary.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).iloc[0]
    center_params = json.loads(best_stage3_row["params_json"])
    best_tree_count = int(best_stage3_row["n_estimators"])

    def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = []
        seen: set[str] = set()
        for candidate in candidates:
            signature = candidate_signature(candidate["params"])
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(candidate)
        return unique

    if stage4_candidates_path.exists():
        stage4_candidates = json.loads(stage4_candidates_path.read_text())
    else:
        params = normalize_candidate(center_params)
        candidates = [candidate_record("stage4_000", params, "stage4_center_from_stage3")]
        max_depth_values = [params.get("base__max_depth")]
        if params.get("base__max_depth") is None:
            max_depth_values.extend([12, 20])
        else:
            max_depth_values.extend([max(6, params["base__max_depth"] - 6), params["base__max_depth"] + 6, None])
        leaf_values = sorted({1, params.get("base__min_samples_leaf", 1), params.get("base__min_samples_leaf", 1) + 1, max(1, params.get("base__min_samples_leaf", 1) - 1)})
        split_values = sorted({2, params.get("base__min_samples_split", 2), params.get("base__min_samples_split", 2) + 2, params.get("base__min_samples_split", 2) + 4})
        feature_values = sorted({round(max(0.15, params.get("model__max_features", 0.5) - 0.10), 2), round(params.get("model__max_features", 0.5), 2), round(min(1.0, params.get("model__max_features", 0.5) + 0.10), 2)})
        sample_values = sorted({round(max(0.40, params.get("model__max_samples", 0.8) - 0.10), 2), round(params.get("model__max_samples", 0.8), 2), round(min(1.0, params.get("model__max_samples", 0.8) + 0.10), 2)})
        criteria_values = [params.get("base__criterion")] + [criterion for criterion in TREE_CRITERIA if criterion != params.get("base__criterion")]
        bootstrap_feature_values = [params.get("model__bootstrap_features"), not params.get("model__bootstrap_features")]

        next_index = 1
        for value in max_depth_values[1:]:
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "base__max_depth": value}, "stage4_depth_probe"))
            next_index += 1
        for value in leaf_values:
            if value == params.get("base__min_samples_leaf"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "base__min_samples_leaf": value}, "stage4_leaf_probe"))
            next_index += 1
        for value in split_values:
            if value == params.get("base__min_samples_split"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "base__min_samples_split": value}, "stage4_split_probe"))
            next_index += 1
        for value in feature_values:
            if value == params.get("model__max_features"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "model__max_features": value}, "stage4_feature_probe"))
            next_index += 1
        for value in sample_values:
            if value == params.get("model__max_samples"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "model__max_samples": value}, "stage4_sample_probe"))
            next_index += 1
        for value in criteria_values:
            if value == params.get("base__criterion"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "base__criterion": value}, "stage4_criterion_probe"))
            next_index += 1
        for value in bootstrap_feature_values:
            if value == params.get("model__bootstrap_features"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "model__bootstrap_features": value}, "stage4_bootstrap_feature_probe"))
            next_index += 1

        stage4_candidates = dedupe_candidates(candidates)
        write_json_atomic(stage4_candidates_path, stage4_candidates)

    stage4_cv = StratifiedKFold(n_splits=profile["stage4_cv"], shuffle=True, random_state=config.random_state)
    stage4_splits = list(stage4_cv.split(X_train, y_train))

    def refresh_stage4_summary() -> pd.DataFrame:
        fold_df = read_dataframe(stage4_fold_path)
        if fold_df.empty:
            return pd.DataFrame()
        summary_rows = []
        for candidate in stage4_candidates:
            candidate_fold_df = fold_df[fold_df["candidate_id"] == candidate["candidate_id"]]
            if len(candidate_fold_df) < profile["stage4_cv"]:
                continue
            summary_rows.append(
                {
                    **candidate_to_row(candidate),
                    "stage": "stage4_local_cv",
                    "n_estimators": best_tree_count,
                    "cv_mean_accuracy": float(candidate_fold_df["fold_accuracy"].mean()),
                    "cv_std_accuracy": float(candidate_fold_df["fold_accuracy"].std(ddof=0)),
                    "total_fit_seconds": float(candidate_fold_df["fit_seconds"].sum()),
                    "completed_folds": int(candidate_fold_df["fold_idx"].nunique()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["cv_mean_accuracy", "total_fit_seconds"], ascending=[False, True]).reset_index(drop=True)
            save_dataframe_atomic(summary_df, stage4_summary_path)
        return summary_df

    stage4_summary = refresh_stage4_summary()
    completed_stage4 = set(stage4_summary["candidate_id"]) if not stage4_summary.empty else set()
    stage_log(f"{MODEL_NAME}: stage4 candidates={len(stage4_candidates)}, completed={len(completed_stage4)}, best_n_estimators={best_tree_count}")

    for candidate in [item for item in stage4_candidates if item["candidate_id"] not in completed_stage4]:
        fold_df = read_dataframe(stage4_fold_path)
        done_folds = set(fold_df.loc[fold_df["candidate_id"] == candidate["candidate_id"], "fold_idx"].astype(int).tolist()) if not fold_df.empty else set()
        for fold_idx, (train_idx, test_idx) in enumerate(stage4_splits):
            if fold_idx in done_folds:
                continue
            start_time = time.time()
            model = build_bagging(
                candidate["params"],
                n_estimators=best_tree_count,
                random_state=config.random_state + 100 + fold_idx,
                n_jobs=ensemble_n_jobs,
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            fold_pred = model.predict(X_train[test_idx])
            fold_accuracy = float((fold_pred == y_train[test_idx]).mean())
            elapsed = time.time() - start_time
            append_rows(
                stage4_fold_path,
                [
                    {
                        **candidate_to_row(candidate),
                        "stage": "stage4_local_cv",
                        "fold_idx": fold_idx,
                        "n_estimators": best_tree_count,
                        "fold_accuracy": fold_accuracy,
                        "fit_seconds": round(elapsed, 3),
                    }
                ],
                sort_cols=["candidate_id", "fold_idx"],
            )
            del model
            gc.collect()
        stage4_summary = refresh_stage4_summary()
        if not stage4_summary.empty:
            update_manifest(
                context,
                {
                    "stage4_completed_candidates": int(stage4_summary["candidate_id"].nunique()),
                    "stage4_total_candidates": len(stage4_candidates),
                    "stage4_best_cv_accuracy": float(stage4_summary["cv_mean_accuracy"].max()),
                },
            )

    stage4_summary = read_dataframe(stage4_summary_path)
    checkpoint_housekeeping(context, "stage4_complete", refresh_bundle=True)

    if not stage1_results.empty:
        plt.figure(figsize=(12, 6))
        top_stage1 = stage1_results.head(12).copy()
        top_stage1["label"] = top_stage1["candidate_id"] + " | " + top_stage1["base__criterion"].astype(str)
        sns.barplot(data=top_stage1, x="oob_accuracy", y="label", palette="crest")
        plt.title("Top Stage 1 bagging candidates by OOB accuracy")
        plt.xlabel("OOB accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage1_oob_top_candidates.png")

    if not stage2_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage2 = stage2_summary.head(12).copy()
        top_stage2["label"] = top_stage2["candidate_id"] + " | " + top_stage2["base__criterion"].astype(str)
        sns.barplot(data=top_stage2, x="cv_mean_accuracy", y="label", palette="mako")
        plt.title("Top Stage 2 bagging candidates by CV accuracy")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage2_cv_top_candidates.png")

    if not stage3_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage3 = stage3_summary.sort_values("cv_mean_accuracy", ascending=False).drop_duplicates("candidate_id").head(5)
        selected_ids = top_stage3["candidate_id"].tolist()
        plot_df = stage3_summary[stage3_summary["candidate_id"].isin(selected_ids)].copy()
        sns.lineplot(data=plot_df, x="n_estimators", y="cv_mean_accuracy", hue="candidate_id", marker="o")
        plt.title("Stage 3 bagging growth curves")
        plt.xlabel("Number of estimators")
        plt.ylabel("Mean CV accuracy")
        save_current_figure(context.output_dir / "stage3_growth_curves.png")

    if not stage4_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage4 = stage4_summary.head(12).copy()
        top_stage4["label"] = top_stage4["candidate_id"] + " | " + top_stage4["source"].astype(str)
        sns.barplot(data=top_stage4, x="cv_mean_accuracy", y="label", palette="rocket")
        plt.title("Top Stage 4 bagging local refinements")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage4_local_refinement.png")

    final_search_table = stage4_summary if not stage4_summary.empty else stage3_summary if not stage3_summary.empty else stage2_summary
    final_stage_name = "stage4_local_cv" if not stage4_summary.empty else "stage3_ensemble_growth" if not stage3_summary.empty else "stage2_cv"
    best_row = final_search_table.sort_values(["cv_mean_accuracy", "total_fit_seconds"] if "total_fit_seconds" in final_search_table.columns else ["cv_mean_accuracy"], ascending=[False, True] if "total_fit_seconds" in final_search_table.columns else [False]).iloc[0]
    best_params = json.loads(best_row["params_json"])
    best_n_estimators = int(best_row["n_estimators"])

    stage_log(f"{MODEL_NAME}: final stage={final_stage_name}, best params={best_params}, n_estimators={best_n_estimators}")

    final_model = build_bagging(best_params, n_estimators=best_n_estimators, random_state=config.random_state + 999, n_jobs=ensemble_n_jobs)
    final_model.fit(X_train, y_train)
    valid_pred = final_model.predict(X_valid)
    validation_report = evaluate_predictions(y_valid, valid_pred)
    render_validation_confusion_matrix(context.output_dir, y_valid, valid_pred, "Validation confusion matrix")

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
        save_current_figure(context.output_dir / "feature_importance_top20.png")

    kaggle_model = build_bagging(best_params, n_estimators=best_n_estimators, random_state=config.random_state + 1001, n_jobs=ensemble_n_jobs)
    kaggle_model.fit(X_full, y_full)
    kaggle_pred = kaggle_model.predict(X_test_full)
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
            "ensemble_n_jobs": ensemble_n_jobs,
            "best_stage": final_stage_name,
            "best_n_estimators": best_n_estimators,
            "best_params": best_params,
            "best_cv_accuracy": float(best_row["cv_mean_accuracy"]),
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
