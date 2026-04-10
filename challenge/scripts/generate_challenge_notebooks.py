from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SUBMISSIONS_DIR = ROOT / "submissions"


COMMON_IMPORTS = dedent(
    """
    from pathlib import Path
    import json
    import warnings

    import matplotlib.pyplot as plt
    import nbformat
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import display
    from sklearn.decomposition import PCA
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.bbox"] = "tight"

    RANDOM_STATE = 301655
    """
).strip()


COMMON_HELPERS = dedent(
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

    TEST_SIZE = 0.20

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


COMMON_EDA = dedent(
    """
    train_df.head()
    """
).strip()


COMMON_EDA_INFO = dedent(
    """
    print("Training shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Sample submission shape:", sample_df.shape)
    print("\\nTarget distribution:")
    display(train_df["class"].value_counts().sort_index())
    print("\\nDtypes summary:")
    display(train_df.dtypes.value_counts())
    """
).strip()


COMMON_QUALITY = dedent(
    """
    quality_report = pd.DataFrame(
        {
            "missing_values": train_df.isna().sum(),
            "missing_pct": train_df.isna().mean().mul(100),
            "n_unique": train_df.nunique(),
        }
    )

    print("Duplicate rows in training:", int(train_df.duplicated().sum()))
    print("Duplicated ids in training:", int(train_df["id"].duplicated().sum()))
    print("Duplicated ids in test:", int(test_df["id"].duplicated().sum()))
    print("Total missing values in training:", int(train_df.isna().sum().sum()))
    print("Total missing values in test:", int(test_df.isna().sum().sum()))
    quality_report.head(10)
    """
).strip()


COMMON_CLASS_PLOT = dedent(
    """
    class_counts = train_df["class"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="viridis")
    plt.title("Class balance in the training set")
    plt.xlabel("Class")
    plt.ylabel("Observations")
    class_balance_path = save_current_figure("class_balance.png")
    plt.show()

    print(f"Saved figure: {class_balance_path}")
    """
).strip()


COMMON_FEATURE_DIFFERENCE = dedent(
    """
    feature_mean_gap = (
        train_df.groupby("class")
        .mean(numeric_only=True)
        .drop(columns=["id"])
        .diff()
        .iloc[-1]
        .abs()
        .sort_values(ascending=False)
    )

    top_gap_features = feature_mean_gap.head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_gap_features.values, y=top_gap_features.index, palette="mako")
    plt.title("Top predictors by absolute difference in class means")
    plt.xlabel("|mean(class=1) - mean(class=0)|")
    plt.ylabel("Predictor")
    gap_path = save_current_figure("top_mean_gaps.png")
    plt.show()

    print(f"Saved figure: {gap_path}")
    display(top_gap_features.to_frame("absolute_mean_gap"))
    """
).strip()


COMMON_BOXPLOTS = dedent(
    """
    top_boxplot_features = top_gap_features.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for axis, feature in zip(axes, top_boxplot_features):
        sns.boxplot(data=train_df, x="class", y=feature, ax=axis, palette="Set2")
        axis.set_title(feature)
        axis.set_xlabel("Class")
        axis.set_ylabel("Value")

    for axis in axes[len(top_boxplot_features):]:
        axis.axis("off")

    boxplot_path = save_current_figure("top_feature_boxplots.png")
    plt.show()

    print(f"Saved figure: {boxplot_path}")
    """
).strip()


COMMON_PCA = dedent(
    """
    scaler_for_pca = StandardScaler()
    X_scaled_full = scaler_for_pca.fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled_full)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        linewidth=2,
    )
    plt.axhline(0.80, color="tomato", linestyle="--", label="80% explained variance")
    plt.axhline(0.90, color="darkgreen", linestyle="--", label="90% explained variance")
    plt.title("Cumulative explained variance from PCA")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    variance_path = save_current_figure("pca_cumulative_variance.png")
    plt.show()

    print(f"Saved figure: {variance_path}")
    print("Components needed for 80% variance:", int(np.argmax(cumulative_variance >= 0.80) + 1))
    print("Components needed for 90% variance:", int(np.argmax(cumulative_variance >= 0.90) + 1))

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X_scaled_full)
    pca_plot_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"]).assign(class_label=y.values)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_plot_df.sample(min(4000, len(pca_plot_df)), random_state=RANDOM_STATE),
        x="PC1",
        y="PC2",
        hue="class_label",
        palette="Set1",
        alpha=0.7,
        s=55,
    )
    plt.title("PCA projection of the training set")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)")
    pca_scatter_path = save_current_figure("pca_scatter.png")
    plt.show()

    print(f"Saved figure: {pca_scatter_path}")
    """
).strip()


COMMON_SPLIT = dedent(
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Training split:", X_train.shape, y_train.shape)
    print("Validation split:", X_valid.shape, y_valid.shape)
    """
).strip()


COMMON_GRID = dedent(
    """
    grid_search = GridSearchCV(
        estimator=MODEL_PIPELINE,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=CV_FOLDS,
        n_jobs=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:")
    print(grid_search.best_params_)
    print("\\nBest cross-validation accuracy:", round(grid_search.best_score_, 4))

    cv_results = (
        pd.DataFrame(grid_search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    display(cv_results.loc[:, ["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10))
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    """
).strip()


COMMON_VALIDATION = dedent(
    """
    valid_predictions = grid_search.predict(X_valid)
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


COMMON_FINAL_MODEL = dedent(
    """
    final_model = grid_search.best_estimator_
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


COMMON_SUMMARY = dedent(
    """
    summary_payload = {
        "model_name": MODEL_NAME,
        "validation_accuracy": validation_accuracy,
        "best_cv_accuracy": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
        "output_dir": str(output_dir),
        "submission_path": str(submission_path),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("Summary saved to:", summary_path)
    summary_payload
    """
).strip()


KNN_MODEL = dedent(
    """
    from sklearn.neighbors import KNeighborsClassifier

    MODEL_NAME = "K-nearest neighbors"
    NOTEBOOK_SLUG = "challenge_01_knn"
    CV_FOLDS = 5

    MODEL_PIPELINE = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier()),
        ]
    )

    PARAM_GRID = {
        "pca__n_components": [30, 40, 50, 60],
        "model__n_neighbors": [5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
    }
    """
).strip()


KNN_PLOT = dedent(
    """
    knn_results = cv_results.copy()
    knn_results["pca__n_components"] = knn_results["param_pca__n_components"].astype(int)
    knn_results["model__n_neighbors"] = knn_results["param_model__n_neighbors"].astype(int)
    knn_results["model__weights"] = knn_results["param_model__weights"].astype(str)

    best_weight = knn_results.iloc[0]["model__weights"]
    heatmap_data = (
        knn_results[knn_results["model__weights"] == best_weight]
        .pivot_table(
            index="pca__n_components",
            columns="model__n_neighbors",
            values="mean_test_score",
        )
        .sort_index()
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"KNN CV accuracy heatmap (weights = {best_weight})")
    plt.xlabel("Number of neighbors")
    plt.ylabel("PCA components")
    heatmap_path = save_current_figure("knn_cv_heatmap.png")
    plt.show()

    print(f"Saved figure: {heatmap_path}")
    """
).strip()


KNN_CONCLUSION = dedent(
    """
    ## Interpretation

    KNN is a distance-based model, so feature scaling is essential.
    PCA is also important here because the dataset has 200 predictors and the raw distance in the original space is noisy.
    The validation heatmap shows how performance changes jointly with the number of principal components and the value of *K*.
    """
).strip()


LOG_MODEL = dedent(
    """
    from sklearn.linear_model import LogisticRegression

    MODEL_NAME = "Logistic regression"
    NOTEBOOK_SLUG = "challenge_02_logistic_regression"
    CV_FOLDS = 5

    MODEL_PIPELINE = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE)),
        ]
    )

    PARAM_GRID = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs"],
    }
    """
).strip()


LOG_PLOT = dedent(
    """
    log_results = cv_results.copy()
    log_results["C"] = log_results["param_model__C"].astype(float)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=log_results, x="C", y="mean_test_score", marker="o")
    plt.xscale("log")
    plt.title("Validation accuracy by regularization strength")
    plt.xlabel("C (inverse regularization)")
    plt.ylabel("Mean CV accuracy")
    c_path = save_current_figure("logreg_c_curve.png")
    plt.show()

    print(f"Saved figure: {c_path}")

    best_logreg = grid_search.best_estimator_
    coefficients = pd.Series(
        best_logreg.named_steps["model"].coef_.ravel(),
        index=X.columns,
    )
    coefficient_plot = pd.concat(
        [
            coefficients.sort_values().head(10),
            coefficients.sort_values(ascending=False).head(10),
        ]
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x=coefficient_plot.values, y=coefficient_plot.index, palette="coolwarm")
    plt.title("Most influential standardized coefficients")
    plt.xlabel("Coefficient value")
    plt.ylabel("Predictor")
    coef_path = save_current_figure("logreg_top_coefficients.png")
    plt.show()

    print(f"Saved figure: {coef_path}")
    """
).strip()


LOG_CONCLUSION = dedent(
    """
    ## Interpretation

    Logistic regression provides a linear decision boundary.
    It is easy to interpret because the sign and magnitude of the coefficients indicate how each standardized predictor moves the decision toward class 0 or class 1.
    """
).strip()


TREE_MODEL = dedent(
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    MODEL_NAME = "Decision tree"
    NOTEBOOK_SLUG = "challenge_03_decision_tree"
    CV_FOLDS = 5

    MODEL_PIPELINE = DecisionTreeClassifier(random_state=RANDOM_STATE)

    PARAM_GRID = {
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }
    """
).strip()


TREE_PLOT = dedent(
    """
    tree_results = cv_results.copy()
    tree_results["max_depth"] = tree_results["param_max_depth"].astype(str)
    tree_results["min_samples_leaf"] = tree_results["param_min_samples_leaf"].astype(int)
    tree_results["criterion"] = tree_results["param_criterion"].astype(str)

    best_criterion = tree_results.iloc[0]["criterion"]
    tree_heatmap = (
        tree_results[tree_results["criterion"] == best_criterion]
        .pivot_table(
            index="max_depth",
            columns="min_samples_leaf",
            values="mean_test_score",
        )
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(tree_heatmap, annot=True, fmt=".3f", cmap="crest")
    plt.title(f"Decision tree CV accuracy heatmap ({best_criterion})")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("max_depth")
    tree_heatmap_path = save_current_figure("tree_cv_heatmap.png")
    plt.show()

    print(f"Saved figure: {tree_heatmap_path}")

    best_tree = grid_search.best_estimator_
    importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Top decision tree feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Predictor")
    importance_path = save_current_figure("tree_feature_importance.png")
    plt.show()

    print(f"Saved figure: {importance_path}")

    plt.figure(figsize=(22, 12))
    plot_tree(
        best_tree,
        feature_names=X.columns,
        class_names=["0", "1"],
        filled=True,
        max_depth=3,
        fontsize=9,
    )
    plt.title("Top levels of the fitted decision tree")
    tree_plot_path = save_current_figure("tree_structure.png")
    plt.show()

    print(f"Saved figure: {tree_plot_path}")
    """
).strip()


TREE_CONCLUSION = dedent(
    """
    ## Interpretation

    The decision tree is easy to debug because every split can be inspected directly.
    The tradeoff is instability: small changes in the data may change the tree structure, which is why pruning and depth control are important.
    """
).strip()


RF_MODEL = dedent(
    """
    from sklearn.ensemble import RandomForestClassifier

    MODEL_NAME = "Random forest"
    NOTEBOOK_SLUG = "challenge_04_random_forest"
    CV_FOLDS = 3

    MODEL_PIPELINE = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    PARAM_GRID = {
        "n_estimators": [100],
        "max_depth": [12, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.3],
    }
    """
).strip()


RF_PLOT = dedent(
    """
    rf_results = cv_results.copy()
    rf_results["max_depth"] = rf_results["param_max_depth"].astype(str)
    rf_results["min_samples_leaf"] = rf_results["param_min_samples_leaf"].astype(int)
    rf_results["max_features"] = rf_results["param_max_features"].astype(str)

    best_max_features = rf_results.iloc[0]["max_features"]
    rf_heatmap = (
        rf_results[rf_results["max_features"] == best_max_features]
        .pivot_table(
            index="max_depth",
            columns="min_samples_leaf",
            values="mean_test_score",
        )
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(rf_heatmap, annot=True, fmt=".3f", cmap="rocket_r")
    plt.title(f"Random forest CV accuracy heatmap (max_features = {best_max_features})")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("max_depth")
    rf_heatmap_path = save_current_figure("rf_cv_heatmap.png")
    plt.show()

    print(f"Saved figure: {rf_heatmap_path}")

    best_rf = grid_search.best_estimator_
    rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=rf_importances.values, y=rf_importances.index, palette="flare")
    plt.title("Top random forest feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Predictor")
    rf_importance_path = save_current_figure("rf_feature_importance.png")
    plt.show()

    print(f"Saved figure: {rf_importance_path}")
    """
).strip()


RF_CONCLUSION = dedent(
    """
    ## Interpretation

    Random forest reduces the variance of a single tree by averaging many trees trained on perturbed versions of the data.
    It is usually more stable than a single tree while preserving a usable measure of feature importance.
    """
).strip()


GB_MODEL = dedent(
    """
    from sklearn.ensemble import GradientBoostingClassifier

    MODEL_NAME = "Gradient boosting"
    NOTEBOOK_SLUG = "challenge_05_gradient_boosting"
    CV_FOLDS = 3

    MODEL_PIPELINE = GradientBoostingClassifier(random_state=RANDOM_STATE)

    PARAM_GRID = {
        "n_estimators": [100],
        "learning_rate": [0.03, 0.05],
        "max_depth": [1, 2],
        "subsample": [0.7, 1.0],
    }
    """
).strip()


GB_PLOT = dedent(
    """
    gb_results = cv_results.copy()
    gb_results["n_estimators"] = gb_results["param_n_estimators"].astype(int)
    gb_results["learning_rate"] = gb_results["param_learning_rate"].astype(float)
    gb_results["max_depth"] = gb_results["param_max_depth"].astype(int)
    gb_results["subsample"] = gb_results["param_subsample"].astype(float)

    best_depth = gb_results.iloc[0]["max_depth"]
    best_subsample = gb_results.iloc[0]["subsample"]
    gb_heatmap = (
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

    plt.figure(figsize=(8, 6))
    sns.heatmap(gb_heatmap, annot=True, fmt=".3f", cmap="magma")
    plt.title(
        f"Gradient boosting CV accuracy heatmap\\nmax_depth = {best_depth}, subsample = {best_subsample}"
    )
    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")
    gb_heatmap_path = save_current_figure("gb_cv_heatmap.png")
    plt.show()

    print(f"Saved figure: {gb_heatmap_path}")

    best_gb = grid_search.best_estimator_
    gb_importances = pd.Series(best_gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=gb_importances.values, y=gb_importances.index, palette="rocket")
    plt.title("Top gradient boosting feature importances")
    plt.xlabel("Importance")
    plt.ylabel("Predictor")
    gb_importance_path = save_current_figure("gb_feature_importance.png")
    plt.show()

    print(f"Saved figure: {gb_importance_path}")
    """
).strip()


GB_CONCLUSION = dedent(
    """
    ## Interpretation

    Gradient boosting builds the final classifier sequentially, correcting the errors of previous trees.
    It is often stronger than a single tree, but it is more sensitive to hyperparameters such as learning rate, depth, subsampling, and the number of estimators.
    """
).strip()


NOTEBOOK_SPECS = [
    {
        "filename": "Challenge_01_KNN.ipynb",
        "title": "# Challenge 01: K-nearest neighbors",
        "model_setup": KNN_MODEL,
        "model_plot": KNN_PLOT,
        "conclusion": KNN_CONCLUSION,
    },
    {
        "filename": "Challenge_02_LogisticRegression.ipynb",
        "title": "# Challenge 02: Logistic regression",
        "model_setup": LOG_MODEL,
        "model_plot": LOG_PLOT,
        "conclusion": LOG_CONCLUSION,
    },
    {
        "filename": "Challenge_03_DecisionTree.ipynb",
        "title": "# Challenge 03: Decision tree",
        "model_setup": TREE_MODEL,
        "model_plot": TREE_PLOT,
        "conclusion": TREE_CONCLUSION,
    },
    {
        "filename": "Challenge_04_RandomForest.ipynb",
        "title": "# Challenge 04: Random forest",
        "model_setup": RF_MODEL,
        "model_plot": RF_PLOT,
        "conclusion": RF_CONCLUSION,
    },
    {
        "filename": "Challenge_05_GradientBoosting.ipynb",
        "title": "# Challenge 05: Gradient boosting",
        "model_setup": GB_MODEL,
        "model_plot": GB_PLOT,
        "conclusion": GB_CONCLUSION,
    },
]


INTRO_MARKDOWN = dedent(
    """
    ## Objective

    The goal is to classify each vibration signal as `0` (undamaged) or `1` (damaged), following the description in `README.md`.
    The notebook follows the same staged style as the reference activities:

    1. Load the data and understand the problem.
    2. Audit data quality.
    3. Explore patterns visually.
    4. Split data into training and validation subsets.
    5. Train and tune one model from the course material.
    6. Evaluate the model using accuracy and a confusion matrix.
    7. Interpret the model and export a Kaggle-style submission.

    All figures generated by this notebook are saved to an individual folder under `challenge/output/`.
    """
).strip()


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook(spec: dict) -> nbf.NotebookNode:
    cells = [
        markdown_cell(spec["title"]),
        markdown_cell(INTRO_MARKDOWN),
        markdown_cell("## 0. Load libraries"),
        code_cell(COMMON_IMPORTS),
        markdown_cell("## 1. Configure the model-specific experiment"),
        code_cell(spec["model_setup"]),
        markdown_cell("## 2. Load the challenge data"),
        code_cell(COMMON_HELPERS),
        markdown_cell("## 3. First look at the training data"),
        code_cell(COMMON_EDA),
        markdown_cell("## 4. Check shape, target balance, and data types"),
        code_cell(COMMON_EDA_INFO),
        markdown_cell("## 5. Data quality audit"),
        code_cell(COMMON_QUALITY),
        markdown_cell("## 6. Plot the class balance"),
        code_cell(COMMON_CLASS_PLOT),
        markdown_cell("## 7. Identify features with the largest mean separation between classes"),
        code_cell(COMMON_FEATURE_DIFFERENCE),
        markdown_cell("## 8. Inspect the six most separated predictors"),
        code_cell(COMMON_BOXPLOTS),
        markdown_cell("## 9. Explore the geometry of the dataset with PCA"),
        code_cell(COMMON_PCA),
        markdown_cell("## 10. Create training and validation splits"),
        code_cell(COMMON_SPLIT),
        markdown_cell("## 11. Train the selected model and tune the hyperparameters"),
        code_cell(COMMON_GRID),
        markdown_cell("## 12. Visualize the tuning behavior for this model"),
        code_cell(spec["model_plot"]),
        markdown_cell("## 13. Evaluate the best model on the validation split"),
        code_cell(COMMON_VALIDATION),
        markdown_cell(spec["conclusion"]),
        markdown_cell("## 14. Refit the best configuration on the full training set and generate submission"),
        code_cell(COMMON_FINAL_MODEL),
        markdown_cell("## 15. Save a compact experiment summary"),
        code_cell(COMMON_SUMMARY),
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in NOTEBOOK_SPECS:
        notebook = build_notebook(spec)
        path = ROOT / spec["filename"]
        nbf.write(notebook, path)
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
