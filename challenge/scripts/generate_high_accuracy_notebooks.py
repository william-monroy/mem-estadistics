from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


COMMON_IMPORTS = dedent(
    """
    from pathlib import Path
    import json
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import display
    from sklearn.decomposition import PCA
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.bbox"] = "tight"

    RANDOM_STATE = 301655
    OUTER_TEST_SIZE = 0.20
    SEARCH_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    """
).strip()


COMMON_DATA = dedent(
    """
    BASE_DIR = Path.cwd()
    TRAIN_PATH = BASE_DIR / "data" / "training.csv"
    TEST_PATH = BASE_DIR / "data" / "test.csv"
    SAMPLE_PATH = BASE_DIR / "data" / "sample.csv"

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_df = pd.read_csv(SAMPLE_PATH)

    X = train_df.drop(columns=["id", "class"])
    y = train_df["class"]
    X_test_kaggle = test_df.drop(columns=["id"])

    output_dir = BASE_DIR / "output" / NOTEBOOK_SLUG
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_dir = BASE_DIR / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    def save_current_figure(filename: str) -> Path:
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        return path

    def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": cm.tolist(),
        }
    """
).strip()


COMMON_OVERVIEW = dedent(
    """
    print("Training shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Sample submission shape:", sample_df.shape)
    print("\\nClass balance:")
    display(train_df["class"].value_counts().sort_index())

    print("\\nMissing values in training:", int(train_df.isna().sum().sum()))
    print("Missing values in test:", int(test_df.isna().sum().sum()))
    print("Duplicated training rows:", int(train_df.duplicated().sum()))
    """
).strip()


COMMON_PCA_DIAG = dedent(
    """
    scaler_for_pca = StandardScaler()
    X_scaled_full = scaler_for_pca.fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled_full)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.axhline(0.80, color="tomato", linestyle="--", label="80% variance")
    plt.axhline(0.90, color="darkgreen", linestyle="--", label="90% variance")
    plt.title("Cumulative explained variance from PCA")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    pca_variance_path = save_current_figure("pca_cumulative_variance.png")
    plt.show()

    print(f"Saved figure: {pca_variance_path}")
    print("Components for 80% variance:", int(np.argmax(cumulative_variance >= 0.80) + 1))
    print("Components for 90% variance:", int(np.argmax(cumulative_variance >= 0.90) + 1))
    """
).strip()


COMMON_SPLIT = dedent(
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=OUTER_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Training split:", X_train.shape, y_train.shape)
    print("Validation split:", X_valid.shape, y_valid.shape)
    """
).strip()


COARSE_SEARCH = dedent(
    """
    coarse_search = GridSearchCV(
        estimator=COARSE_PIPELINE,
        param_grid=COARSE_GRID,
        scoring="accuracy",
        cv=SEARCH_CV,
        n_jobs=1,
        refit=True,
        verbose=2,
    )

    coarse_search.fit(X_train, y_train)

    print("Coarse best CV accuracy:", round(coarse_search.best_score_, 4))
    print("Coarse best params:")
    print(coarse_search.best_params_)

    coarse_results = (
        pd.DataFrame(coarse_search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    display(coarse_results.loc[:, ["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10))
    coarse_results.to_csv(output_dir / "coarse_cv_results.csv", index=False)
    """
).strip()


FINE_SEARCH = dedent(
    """
    fine_search = GridSearchCV(
        estimator=FINE_PIPELINE,
        param_grid=FINE_GRID,
        scoring="accuracy",
        cv=SEARCH_CV,
        n_jobs=1,
        refit=True,
        verbose=2,
    )

    fine_search.fit(X_train, y_train)

    print("Fine best CV accuracy:", round(fine_search.best_score_, 4))
    print("Fine best params:")
    print(fine_search.best_params_)

    fine_results = (
        pd.DataFrame(fine_search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    display(fine_results.loc[:, ["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10))
    fine_results.to_csv(output_dir / "fine_cv_results.csv", index=False)

    search = fine_search
    """
).strip()


VALIDATION = dedent(
    """
    valid_predictions = search.predict(X_valid)
    validation_report = evaluate_predictions(y_valid, valid_predictions)
    validation_accuracy = validation_report["accuracy"]

    print("Validation accuracy:", round(validation_accuracy, 4))
    print("Confusion matrix:")
    print(np.array(validation_report["confusion_matrix"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_valid,
        valid_predictions,
        display_labels=["Undamaged (0)", "Damaged (1)"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title("Validation confusion matrix")
    confusion_path = save_current_figure("validation_confusion_matrix.png")
    plt.show()

    print(f"Saved figure: {confusion_path}")
    """
).strip()


FINAL_MODEL = dedent(
    """
    final_model = search.best_estimator_
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test_kaggle)

    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "class": test_predictions.astype(int),
        }
    )

    submission_path = submission_dir / f"{NOTEBOOK_SLUG}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    submission_df.head()

    print(f"Submission saved to: {submission_path}")
    """
).strip()


SUMMARY = dedent(
    """
    summary_payload = {
        "model_name": MODEL_NAME,
        "strategy": "coarse_to_fine_grid_search",
        "baseline_reference": BASELINE_REFERENCE,
        "coarse_best_cv_accuracy": float(coarse_search.best_score_),
        "coarse_best_params": coarse_search.best_params_,
        "fine_best_cv_accuracy": float(fine_search.best_score_),
        "fine_best_params": fine_search.best_params_,
        "validation_accuracy": validation_accuracy,
        "output_dir": str(output_dir),
        "submission_path": str(submission_path),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("Summary saved to:", summary_path)
    summary_payload
    """
).strip()


KNN_SETUP = dedent(
    """
    from sklearn.neighbors import KNeighborsClassifier

    MODEL_NAME = "K-nearest neighbors"
    NOTEBOOK_SLUG = "challenge_01_knn_tuned"
    BASELINE_REFERENCE = {
        "previous_local_best_validation_accuracy": 0.8285,
        "goal": "Push KNN beyond the current local baseline and explore whether 0.90 is reachable."
    }

    COARSE_PIPELINE = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier()),
        ]
    )

    COARSE_GRID = {
        "pca__n_components": [36, 40, 44, 48, 52, 56, 60],
        "model__n_neighbors": [1, 2, 3, 4, 5, 6, 8, 10],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["manhattan", "euclidean", "chebyshev"],
    }
    """
).strip()


KNN_FINE_GRID = dedent(
    """
    best_components = coarse_search.best_params_["pca__n_components"]
    best_k = coarse_search.best_params_["model__n_neighbors"]
    best_metric = coarse_search.best_params_["model__metric"]
    best_weights = coarse_search.best_params_["model__weights"]

    candidate_components = sorted(
        {
            value
            for value in [
                best_components - 4,
                best_components - 2,
                best_components - 1,
                best_components,
                best_components + 1,
                best_components + 2,
                best_components + 4,
            ]
            if 12 <= value <= X_train.shape[1]
        }
    )

    candidate_neighbors = sorted(
        {
            value
            for value in [
                best_k - 2,
                best_k - 1,
                best_k,
                best_k + 1,
                best_k + 2,
                best_k + 4,
            ]
            if value >= 1
        }
    )

    candidate_metrics = sorted({best_metric, "manhattan", "euclidean"})
    candidate_weights = sorted({best_weights, "uniform", "distance"})

    FINE_PIPELINE = COARSE_PIPELINE
    FINE_GRID = {
        "pca__n_components": candidate_components,
        "model__n_neighbors": candidate_neighbors,
        "model__weights": candidate_weights,
        "model__metric": candidate_metrics,
    }

    print("Fine grid:")
    print(FINE_GRID)
    """
).strip()


KNN_PLOT = dedent(
    """
    knn_results = fine_results.copy()
    knn_results["pca__n_components"] = knn_results["param_pca__n_components"].astype(int)
    knn_results["model__n_neighbors"] = knn_results["param_model__n_neighbors"].astype(int)
    knn_results["model__weights"] = knn_results["param_model__weights"].astype(str)
    knn_results["model__metric"] = knn_results["param_model__metric"].astype(str)

    best_metric = knn_results.iloc[0]["model__metric"]
    best_weight = knn_results.iloc[0]["model__weights"]

    heatmap_data = (
        knn_results[
            (knn_results["model__metric"] == best_metric)
            & (knn_results["model__weights"] == best_weight)
        ]
        .pivot_table(
            index="pca__n_components",
            columns="model__n_neighbors",
            values="mean_test_score",
        )
        .sort_index()
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"KNN fine-search heatmap\\nmetric={best_metric}, weights={best_weight}")
    plt.xlabel("Number of neighbors")
    plt.ylabel("PCA components")
    heatmap_path = save_current_figure("knn_fine_heatmap.png")
    plt.show()

    print(f"Saved figure: {heatmap_path}")
    """
).strip()


KNN_INTERPRET = dedent(
    """
    ## Interpretation

    KNN remains the strongest family found so far.
    This notebook intentionally concentrates the search around the most promising region: low-to-mid numbers of neighbors, PCA in the 40-60 range, and distance metrics that performed better in previous runs.
    """
).strip()


RF_SETUP = dedent(
    """
    from sklearn.ensemble import RandomForestClassifier

    MODEL_NAME = "Random forest"
    NOTEBOOK_SLUG = "challenge_04_random_forest_tuned"
    BASELINE_REFERENCE = {
        "previous_local_best_validation_accuracy": 0.7405,
        "goal": "Test whether larger forests and broader split settings can materially close the gap with KNN."
    }

    COARSE_PIPELINE = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    COARSE_GRID = {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [None, 24, 36],
        "min_samples_leaf": [1, 2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "max_features": [0.2, 0.3, 0.5, "sqrt"],
        "criterion": ["gini", "entropy"],
    }
    """
).strip()


RF_FINE_GRID = dedent(
    """
    best_n_estimators = coarse_search.best_params_["n_estimators"]
    best_max_depth = coarse_search.best_params_["max_depth"]
    best_min_samples_leaf = coarse_search.best_params_["min_samples_leaf"]
    best_min_samples_split = coarse_search.best_params_["min_samples_split"]
    best_max_features = coarse_search.best_params_["max_features"]
    best_criterion = coarse_search.best_params_["criterion"]

    candidate_n_estimators = sorted(
        {
            value
            for value in [
                max(100, best_n_estimators // 2),
                best_n_estimators,
                best_n_estimators + 200,
                best_n_estimators + 400,
            ]
            if value <= 1200
        }
    )

    depth_candidates = {best_max_depth, None}
    if isinstance(best_max_depth, int):
        depth_candidates.update({max(8, best_max_depth - 8), best_max_depth + 8})

    feature_candidates = {best_max_features, "sqrt"}
    if isinstance(best_max_features, float):
        feature_candidates.update(
            {
                max(0.1, round(best_max_features - 0.1, 2)),
                round(best_max_features, 2),
                min(0.8, round(best_max_features + 0.1, 2)),
            }
        )
    else:
        feature_candidates.update({0.3, 0.5})

    FINE_PIPELINE = COARSE_PIPELINE
    FINE_GRID = {
        "n_estimators": candidate_n_estimators,
        "max_depth": list(depth_candidates),
        "min_samples_leaf": sorted({1, best_min_samples_leaf, best_min_samples_leaf + 1, best_min_samples_leaf + 2}),
        "min_samples_split": sorted({2, best_min_samples_split, best_min_samples_split + 5}),
        "max_features": list(feature_candidates),
        "criterion": sorted({best_criterion, "gini", "entropy", "log_loss"}),
    }

    print("Fine grid:")
    print(FINE_GRID)
    """
).strip()


RF_PLOT = dedent(
    """
    rf_results = fine_results.copy()
    rf_results["n_estimators"] = rf_results["param_n_estimators"].astype(int)
    rf_results["max_depth"] = rf_results["param_max_depth"].astype(str)
    rf_results["min_samples_leaf"] = rf_results["param_min_samples_leaf"].astype(int)
    rf_results["max_features"] = rf_results["param_max_features"].astype(str)

    best_depth = rf_results.iloc[0]["max_depth"]
    best_leaf = rf_results.iloc[0]["min_samples_leaf"]

    heatmap_data = (
        rf_results[
            (rf_results["max_depth"] == best_depth)
            & (rf_results["min_samples_leaf"] == best_leaf)
        ]
        .pivot_table(
            index="max_features",
            columns="n_estimators",
            values="mean_test_score",
        )
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="rocket_r")
    plt.title(f"Random forest fine-search heatmap\\nmax_depth={best_depth}, min_samples_leaf={best_leaf}")
    plt.xlabel("Number of trees")
    plt.ylabel("max_features")
    heatmap_path = save_current_figure("rf_fine_heatmap.png")
    plt.show()

    print(f"Saved figure: {heatmap_path}")

    importances = pd.Series(search.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances.values, y=importances.index, palette="flare")
    plt.title("Top random forest feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Predictor")
    importance_path = save_current_figure("rf_feature_importance.png")
    plt.show()

    print(f"Saved figure: {importance_path}")
    """
).strip()


RF_INTERPRET = dedent(
    """
    ## Interpretation

    Random forest is being pushed here with larger forests, wider feature-subsampling ranges, and looser tree-growth controls than the previous notebook.
    If it still stays well below KNN after this search, that is strong evidence that this family is not the best use of training time for this dataset.
    """
).strip()


GB_SETUP = dedent(
    """
    from sklearn.ensemble import GradientBoostingClassifier

    MODEL_NAME = "Gradient boosting"
    NOTEBOOK_SLUG = "challenge_05_gradient_boosting_tuned"
    BASELINE_REFERENCE = {
        "previous_local_best_validation_accuracy": 0.6710,
        "goal": "Probe whether a slower, deeper boosting configuration can recover a strong nonlinear decision boundary."
    }

    COARSE_PIPELINE = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )

    COARSE_GRID = {
        "n_estimators": [100, 200, 400, 600],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [1, 2, 4, 8],
        "subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
        "max_features": [None, 0.5],
    }
    """
).strip()


GB_FINE_GRID = dedent(
    """
    best_n_estimators = coarse_search.best_params_["n_estimators"]
    best_learning_rate = coarse_search.best_params_["learning_rate"]
    best_max_depth = coarse_search.best_params_["max_depth"]
    best_min_samples_leaf = coarse_search.best_params_["min_samples_leaf"]
    best_subsample = coarse_search.best_params_["subsample"]
    best_max_features = coarse_search.best_params_["max_features"]

    candidate_estimators = sorted(
        {
            value
            for value in [
                max(100, best_n_estimators // 2),
                best_n_estimators,
                best_n_estimators + 100,
                best_n_estimators + 300,
            ]
            if value <= 1000
        }
    )

    candidate_learning_rates = sorted(
        {
            round(value, 3)
            for value in [
                best_learning_rate / 2,
                best_learning_rate * 0.75,
                best_learning_rate,
                best_learning_rate * 1.25,
                best_learning_rate * 1.5,
            ]
            if 0.005 <= value <= 0.12
        }
    )

    candidate_subsamples = sorted(
        {
            round(value, 2)
            for value in [
                max(0.4, best_subsample - 0.1),
                best_subsample,
                min(1.0, best_subsample + 0.1),
            ]
        }
    )

    FINE_PIPELINE = COARSE_PIPELINE
    FINE_GRID = {
        "n_estimators": candidate_estimators,
        "learning_rate": candidate_learning_rates,
        "max_depth": sorted({best_max_depth - 1, best_max_depth, best_max_depth + 1} - {0}),
        "min_samples_leaf": sorted({1, best_min_samples_leaf, best_min_samples_leaf + 1, best_min_samples_leaf + 2}),
        "subsample": candidate_subsamples,
        "max_features": list({best_max_features, None, 0.5}),
    }

    print("Fine grid:")
    print(FINE_GRID)
    """
).strip()


GB_PLOT = dedent(
    """
    gb_results = fine_results.copy()
    gb_results["n_estimators"] = gb_results["param_n_estimators"].astype(int)
    gb_results["learning_rate"] = gb_results["param_learning_rate"].astype(float)
    gb_results["max_depth"] = gb_results["param_max_depth"].astype(int)
    gb_results["subsample"] = gb_results["param_subsample"].astype(float)

    best_depth = gb_results.iloc[0]["max_depth"]
    best_subsample = gb_results.iloc[0]["subsample"]

    heatmap_data = (
        gb_results[
            (gb_results["max_depth"] == best_depth)
            & (gb_results["subsample"] == best_subsample)
        ]
        .pivot_table(
            index="learning_rate",
            columns="n_estimators",
            values="mean_test_score",
        )
        .sort_index()
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="magma")
    plt.title(f"Gradient boosting fine-search heatmap\\nmax_depth={best_depth}, subsample={best_subsample}")
    plt.xlabel("Number of estimators")
    plt.ylabel("Learning rate")
    heatmap_path = save_current_figure("gb_fine_heatmap.png")
    plt.show()

    print(f"Saved figure: {heatmap_path}")

    importances = pd.Series(search.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances.values, y=importances.index, palette="rocket")
    plt.title("Top gradient boosting feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Predictor")
    importance_path = save_current_figure("gb_feature_importance.png")
    plt.show()

    print(f"Saved figure: {importance_path}")
    """
).strip()


GB_INTERPRET = dedent(
    """
    ## Interpretation

    Gradient boosting is being searched here with slower learning rates, deeper weak learners, and broader stochasticity controls.
    This is intentionally expensive. If it still lags after this notebook, it is unlikely to be the winning family for this challenge.
    """
).strip()


NOTEBOOK_SPECS = [
    {
        "filename": "Challenge_01_KNN_Tuned.ipynb",
        "title": "# Challenge 01 Tuned: K-nearest neighbors",
        "setup": KNN_SETUP,
        "fine_grid": KNN_FINE_GRID,
        "plot": KNN_PLOT,
        "interpretation": KNN_INTERPRET,
    },
    {
        "filename": "Challenge_04_RandomForest_Tuned.ipynb",
        "title": "# Challenge 04 Tuned: Random forest",
        "setup": RF_SETUP,
        "fine_grid": RF_FINE_GRID,
        "plot": RF_PLOT,
        "interpretation": RF_INTERPRET,
    },
    {
        "filename": "Challenge_05_GradientBoosting_Tuned.ipynb",
        "title": "# Challenge 05 Tuned: Gradient boosting",
        "setup": GB_SETUP,
        "fine_grid": GB_FINE_GRID,
        "plot": GB_PLOT,
        "interpretation": GB_INTERPRET,
    },
]


INTRO = dedent(
    """
    ## Objective

    This notebook is intentionally designed for a heavier search budget.
    The goal is to push the selected model beyond the previous local baseline using a two-stage search:

    1. A broad coarse search across the most promising region.
    2. A fine search built automatically around the best coarse configuration.

    These notebooks are expected to take substantially longer than the `Ready` versions.
    They are prepared for manual execution and are not guaranteed to exceed 0.90, but they are explicitly configured to chase that target as hard as this model family reasonably allows.
    """
).strip()


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook(spec: dict) -> nbf.NotebookNode:
    cells = [
        markdown_cell(spec["title"]),
        markdown_cell(INTRO),
        markdown_cell("## 0. Load libraries"),
        code_cell(COMMON_IMPORTS),
        markdown_cell("## 1. Configure the model family and the coarse search space"),
        code_cell(spec["setup"]),
        markdown_cell("## 2. Load the challenge data"),
        code_cell(COMMON_DATA),
        markdown_cell("## 3. Quick audit of the dataset"),
        code_cell(COMMON_OVERVIEW),
        markdown_cell("## 4. PCA diagnostic"),
        code_cell(COMMON_PCA_DIAG),
        markdown_cell("## 5. Create training and validation splits"),
        code_cell(COMMON_SPLIT),
        markdown_cell("## 6. Coarse search"),
        code_cell(COARSE_SEARCH),
        markdown_cell("## 7. Build the fine search space around the best coarse configuration"),
        code_cell(spec["fine_grid"]),
        markdown_cell("## 8. Fine search"),
        code_cell(FINE_SEARCH),
        markdown_cell("## 9. Visualize the fine-search surface"),
        code_cell(spec["plot"]),
        markdown_cell("## 10. Evaluate the best tuned model on the holdout validation split"),
        code_cell(VALIDATION),
        markdown_cell(spec["interpretation"]),
        markdown_cell("## 11. Refit on the full training set and generate the submission file"),
        code_cell(FINAL_MODEL),
        markdown_cell("## 12. Save the tuning summary"),
        code_cell(SUMMARY),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }
    return notebook


def main() -> None:
    for spec in NOTEBOOK_SPECS:
        notebook = build_notebook(spec)
        path = ROOT / spec["filename"]
        nbf.write(notebook, path)
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
