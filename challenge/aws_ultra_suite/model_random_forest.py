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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

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


MODEL_KEY = "random_forest"
MODEL_NAME = "Random forest"
SLUG = "challenge_04_random_forest_aws_ultra"


SEARCH_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "stage1_n_iter": 72,
        "stage1_train_fraction": 0.70,
        "stage1_n_estimators": 192,
        "stage2_top_k": 16,
        "stage2_cv": 3,
        "stage2_n_estimators": 384,
        "stage3_top_k": 4,
        "stage3_cv": 5,
        "stage3_tree_schedule": [384, 640, 896],
        "stage4_cv": 5,
    },
    "aws_fast": {
        "stage1_n_iter": 112,
        "stage1_train_fraction": 0.85,
        "stage1_n_estimators": 256,
        "stage2_top_k": 20,
        "stage2_cv": 3,
        "stage2_n_estimators": 512,
        "stage3_top_k": 5,
        "stage3_cv": 5,
        "stage3_tree_schedule": [512, 768, 1024, 1280],
        "stage4_cv": 5,
    },
    "aws_max": {
        "stage1_n_iter": 144,
        "stage1_train_fraction": 0.90,
        "stage1_n_estimators": 320,
        "stage2_top_k": 24,
        "stage2_cv": 3,
        "stage2_n_estimators": 640,
        "stage3_top_k": 6,
        "stage3_cv": 5,
        "stage3_tree_schedule": [640, 960, 1280, 1600],
        "stage4_cv": 5,
    },
}


def get_supported_criteria() -> list[str]:
    version_parts = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    criteria = ["gini", "entropy"]
    if version_parts >= (1, 1):
        criteria.append("log_loss")
    return criteria


RF_CRITERIA = get_supported_criteria()


def normalize_candidate(params: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: normalize_value(value) for key, value in params.items()}
    if normalized.get("max_depth") is not None:
        normalized["max_depth"] = int(normalized["max_depth"])
    if normalized.get("min_samples_leaf") is not None:
        normalized["min_samples_leaf"] = int(normalized["min_samples_leaf"])
    if normalized.get("min_samples_split") is not None:
        normalized["min_samples_split"] = int(normalized["min_samples_split"])
    if normalized.get("max_features") not in {None, "sqrt", "log2"}:
        normalized["max_features"] = float(normalized["max_features"])
    if normalized.get("max_samples") is not None:
        normalized["max_samples"] = float(normalized["max_samples"])
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
        "criterion": params.get("criterion"),
        "max_depth": params.get("max_depth"),
        "min_samples_leaf": params.get("min_samples_leaf"),
        "min_samples_split": params.get("min_samples_split"),
        "max_features": params.get("max_features"),
        "max_samples": params.get("max_samples"),
        "params_json": json.dumps(params, sort_keys=True),
    }


def build_rf(params: dict[str, Any], *, n_estimators: int, random_state: int, n_jobs: int, oob_score: bool = False, warm_start: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        bootstrap=True,
        oob_score=oob_score,
        warm_start=warm_start,
        n_jobs=n_jobs,
        random_state=random_state,
        **normalize_candidate(params),
    )


def run(config: AwsUltraConfig) -> Path:
    profile = SEARCH_PRESETS.get(config.profile, SEARCH_PRESETS["aws_fast"])
    context = build_model_context(config, MODEL_KEY, SLUG, MODEL_NAME)
    ensemble_n_jobs = config.resolved_ensemble_n_jobs()

    stage_log(f"{MODEL_NAME}: starting with profile={config.profile}")
    stage_log(f"{MODEL_NAME}: runtime snapshot -> {cpu_runtime_snapshot(config)}")
    stage_log(f"{MODEL_NAME}: n_jobs={ensemble_n_jobs}, criteria={RF_CRITERIA}")

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
        "criterion": RF_CRITERIA,
        "max_depth": [None, 16, 24, 32, 40, 56],
        "min_samples_leaf": [1, 2, 3, 4, 6],
        "min_samples_split": [2, 4, 6, 10, 14],
        "max_features": ["sqrt", 0.12, 0.18, 0.25, 0.35, 0.50, 0.70],
        "max_samples": [None, 0.65, 0.80, 0.90],
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
        model = build_rf(
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
            model = build_rf(
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
            grouped["criterion"] = candidate["params"].get("criterion")
            grouped["max_depth"] = candidate["params"].get("max_depth")
            grouped["min_samples_leaf"] = candidate["params"].get("min_samples_leaf")
            grouped["min_samples_split"] = candidate["params"].get("min_samples_split")
            grouped["max_features"] = candidate["params"].get("max_features")
            grouped["max_samples"] = candidate["params"].get("max_samples")
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
            model = build_rf(
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

        max_depth_values = [params.get("max_depth")]
        if params.get("max_depth") is None:
            max_depth_values.extend([24, 40])
        else:
            max_depth_values.extend([max(8, params["max_depth"] - 8), params["max_depth"] + 8, None])

        leaf_values = sorted({1, params.get("min_samples_leaf", 1), params.get("min_samples_leaf", 1) + 1, max(1, params.get("min_samples_leaf", 1) - 1)})
        split_values = sorted({2, params.get("min_samples_split", 2), params.get("min_samples_split", 2) + 2, params.get("min_samples_split", 2) + 4})
        max_features = params.get("max_features")
        if isinstance(max_features, float):
            feature_values = sorted({round(max(0.10, max_features - 0.05), 2), round(max_features, 2), round(min(0.80, max_features + 0.05), 2)})
            feature_values.append("sqrt")
        else:
            feature_values = ["sqrt", 0.20, 0.30, 0.50]
        max_samples = params.get("max_samples")
        if max_samples is None:
            sample_values = [None, 0.70, 0.85]
        else:
            sample_values = sorted({round(max(0.55, max_samples - 0.10), 2), round(max_samples, 2), round(min(0.95, max_samples + 0.10), 2), None}, key=lambda x: (x is None, x))
        criteria_values = [params.get("criterion")] + [criterion for criterion in RF_CRITERIA if criterion != params.get("criterion")]

        next_index = 1
        for value in max_depth_values[1:]:
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "max_depth": value}, "stage4_depth_probe"))
            next_index += 1
        for value in leaf_values:
            if value == params.get("min_samples_leaf"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "min_samples_leaf": value}, "stage4_leaf_probe"))
            next_index += 1
        for value in split_values:
            if value == params.get("min_samples_split"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "min_samples_split": value}, "stage4_split_probe"))
            next_index += 1
        for value in feature_values:
            if value == params.get("max_features"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "max_features": value}, "stage4_feature_probe"))
            next_index += 1
        for value in sample_values:
            if value == params.get("max_samples"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "max_samples": value}, "stage4_sample_probe"))
            next_index += 1
        for value in criteria_values:
            if value == params.get("criterion"):
                continue
            candidates.append(candidate_record(f"stage4_{next_index:03d}", {**params, "criterion": value}, "stage4_criterion_probe"))
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
    stage_log(f"{MODEL_NAME}: stage4 candidates={len(stage4_candidates)}, completed={len(completed_stage4)}, best_tree_count={best_tree_count}")

    for candidate in [item for item in stage4_candidates if item["candidate_id"] not in completed_stage4]:
        fold_df = read_dataframe(stage4_fold_path)
        done_folds = set(fold_df.loc[fold_df["candidate_id"] == candidate["candidate_id"], "fold_idx"].astype(int).tolist()) if not fold_df.empty else set()
        for fold_idx, (train_idx, test_idx) in enumerate(stage4_splits):
            if fold_idx in done_folds:
                continue
            start_time = time.time()
            model = build_rf(
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
        top_stage1["label"] = top_stage1["candidate_id"] + " | " + top_stage1["criterion"].astype(str)
        sns.barplot(data=top_stage1, x="oob_accuracy", y="label", palette="crest")
        plt.title("Top Stage 1 random forest candidates by OOB accuracy")
        plt.xlabel("OOB accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage1_oob_top_candidates.png")

    if not stage2_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage2 = stage2_summary.head(12).copy()
        top_stage2["label"] = top_stage2["candidate_id"] + " | " + top_stage2["criterion"].astype(str)
        sns.barplot(data=top_stage2, x="cv_mean_accuracy", y="label", palette="mako")
        plt.title("Top Stage 2 random forest candidates by CV accuracy")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage2_cv_top_candidates.png")

    if not stage3_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage3 = stage3_summary.sort_values("cv_mean_accuracy", ascending=False).drop_duplicates("candidate_id").head(5)
        selected_ids = top_stage3["candidate_id"].tolist()
        plot_df = stage3_summary[stage3_summary["candidate_id"].isin(selected_ids)].copy()
        sns.lineplot(data=plot_df, x="n_estimators", y="cv_mean_accuracy", hue="candidate_id", marker="o")
        plt.title("Stage 3 random forest growth curves")
        plt.xlabel("Number of trees")
        plt.ylabel("Mean CV accuracy")
        save_current_figure(context.output_dir / "stage3_growth_curves.png")

    if not stage4_summary.empty:
        plt.figure(figsize=(12, 6))
        top_stage4 = stage4_summary.head(12).copy()
        top_stage4["label"] = top_stage4["candidate_id"] + " | " + top_stage4["source"].astype(str)
        sns.barplot(data=top_stage4, x="cv_mean_accuracy", y="label", palette="rocket")
        plt.title("Top Stage 4 random forest local refinements")
        plt.xlabel("Mean CV accuracy")
        plt.ylabel("Candidate")
        save_current_figure(context.output_dir / "stage4_local_refinement.png")

    final_search_table = stage4_summary if not stage4_summary.empty else stage3_summary if not stage3_summary.empty else stage2_summary
    final_stage_name = "stage4_local_cv" if not stage4_summary.empty else "stage3_ensemble_growth" if not stage3_summary.empty else "stage2_cv"
    best_row = final_search_table.sort_values(["cv_mean_accuracy", "total_fit_seconds"] if "total_fit_seconds" in final_search_table.columns else ["cv_mean_accuracy"], ascending=[False, True] if "total_fit_seconds" in final_search_table.columns else [False]).iloc[0]
    best_params = json.loads(best_row["params_json"])
    best_n_estimators = int(best_row["n_estimators"])

    stage_log(f"{MODEL_NAME}: final stage={final_stage_name}, best params={best_params}, n_estimators={best_n_estimators}")

    final_model = build_rf(best_params, n_estimators=best_n_estimators, random_state=config.random_state + 999, n_jobs=ensemble_n_jobs)
    final_model.fit(X_train, y_train)
    valid_pred = final_model.predict(X_valid)
    validation_report = evaluate_predictions(y_valid, valid_pred)
    render_validation_confusion_matrix(context.output_dir, y_valid, valid_pred, "Validation confusion matrix")

    importances = pd.Series(final_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Top 20 feature importances from final random forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_current_figure(context.output_dir / "feature_importance_top20.png")

    kaggle_model = build_rf(best_params, n_estimators=best_n_estimators, random_state=config.random_state + 1001, n_jobs=ensemble_n_jobs)
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
