from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

from generate_challenge_notebooks import (
    COMMON_BOXPLOTS,
    COMMON_CLASS_PLOT,
    COMMON_EDA,
    COMMON_EDA_INFO,
    COMMON_FEATURE_DIFFERENCE,
    COMMON_HELPERS,
    COMMON_IMPORTS,
    COMMON_PCA,
    COMMON_QUALITY,
    COMMON_SPLIT,
    GB_CONCLUSION,
    LOG_CONCLUSION,
    RF_CONCLUSION,
    ROOT,
    TREE_CONCLUSION,
)


READY_INTRO = dedent(
    """
    ## Objective

    This notebook is the fast-to-run version of the corresponding challenge model.
    The hyperparameter search was already done previously, so here the model is configured with the best parameters found and is ready for manual execution.

    Workflow:

    1. Load and audit the data.
    2. Explore key patterns graphically.
    3. Split the data into training and validation subsets.
    4. Train the model with fixed best hyperparameters.
    5. Evaluate the model on the validation split.
    6. Interpret the model.
    7. Refit on the full training set and export a Kaggle-style submission.
    """
).strip()


READY_FIT = dedent(
    """
    print("Fixed best hyperparameters:")
    print(BEST_PARAMS)

    fitted_model = MODEL_PIPELINE
    fitted_model.fit(X_train, y_train)

    print("Model fitted successfully.")
    """
).strip()


READY_VALIDATION = dedent(
    """
    valid_predictions = fitted_model.predict(X_valid)
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


READY_FINAL_MODEL = dedent(
    """
    final_model = MODEL_PIPELINE
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


READY_SUMMARY = dedent(
    """
    summary_payload = {
        "model_name": MODEL_NAME,
        "mode": "fixed_best_params",
        "validation_accuracy": validation_accuracy,
        "best_params": BEST_PARAMS,
        "output_dir": str(output_dir),
        "submission_path": str(submission_path),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("Summary saved to:", summary_path)
    summary_payload
    """
).strip()


KNN_READY_MODEL = dedent(
    """
    from sklearn.neighbors import KNeighborsClassifier

    MODEL_NAME = "K-nearest neighbors"
    NOTEBOOK_SLUG = "challenge_01_knn_ready"

    BEST_PARAMS = {
        "pca__n_components": 48,
        "model__n_neighbors": 4,
        "model__weights": "distance",
        "model__metric": "manhattan",
    }

    MODEL_PIPELINE = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=48, random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier(n_neighbors=4, weights="distance", metric="manhattan")),
        ]
    )
    """
).strip()


KNN_READY_PLOT = dedent(
    """
    X_valid_scaled = fitted_model.named_steps["scaler"].transform(X_valid)
    X_valid_pca = fitted_model.named_steps["pca"].transform(X_valid_scaled)

    validation_projection = pd.DataFrame(
        X_valid_pca[:, :2],
        columns=["PC1", "PC2"],
    ).assign(
        actual=y_valid.values,
        predicted=valid_predictions,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(
        data=validation_projection,
        x="PC1",
        y="PC2",
        hue="actual",
        palette="Set1",
        alpha=0.7,
        s=50,
        ax=axes[0],
    )
    axes[0].set_title("Validation set projected with the fitted PCA\\nColored by actual class")

    sns.scatterplot(
        data=validation_projection,
        x="PC1",
        y="PC2",
        hue="predicted",
        palette="Set2",
        alpha=0.7,
        s=50,
        ax=axes[1],
    )
    axes[1].set_title("Validation set projected with the fitted PCA\\nColored by predicted class")

    knn_plot_path = save_current_figure("knn_validation_projection.png")
    plt.show()

    print(f"Saved figure: {knn_plot_path}")
    """
).strip()


KNN_READY_CONCLUSION = dedent(
    """
    ## Interpretation

    This fixed KNN configuration uses standardized predictors, 50 principal components, and distance-weighted neighbors with `K = 5`.
    It is the strongest configuration found in the previous search, so this notebook is ready to be executed directly without a new tuning phase.
    """
).strip()


LOG_READY_MODEL = dedent(
    """
    from sklearn.linear_model import LogisticRegression

    MODEL_NAME = "Logistic regression"
    NOTEBOOK_SLUG = "challenge_02_logistic_regression_ready"

    BEST_PARAMS = {
        "model__C": 1.0,
        "model__solver": "lbfgs",
    }

    MODEL_PIPELINE = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, random_state=RANDOM_STATE)),
        ]
    )
    """
).strip()


LOG_READY_PLOT = dedent(
    """
    coefficients = pd.Series(
        fitted_model.named_steps["model"].coef_.ravel(),
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


TREE_READY_MODEL = dedent(
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    MODEL_NAME = "Decision tree"
    NOTEBOOK_SLUG = "challenge_03_decision_tree_ready"

    BEST_PARAMS = {
        "criterion": "gini",
        "max_depth": 12,
        "min_samples_leaf": 5,
    }

    MODEL_PIPELINE = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    """
).strip()


TREE_READY_PLOT = dedent(
    """
    importances = pd.Series(fitted_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

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
        fitted_model,
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


RF_READY_MODEL = dedent(
    """
    from sklearn.ensemble import RandomForestClassifier

    MODEL_NAME = "Random forest"
    NOTEBOOK_SLUG = "challenge_04_random_forest_ready"

    BEST_PARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    }

    MODEL_PIPELINE = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    """
).strip()


RF_READY_PLOT = dedent(
    """
    rf_importances = pd.Series(fitted_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

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


GB_READY_MODEL = dedent(
    """
    from sklearn.ensemble import GradientBoostingClassifier

    MODEL_NAME = "Gradient boosting"
    NOTEBOOK_SLUG = "challenge_05_gradient_boosting_ready"

    BEST_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.6,
    }

    MODEL_PIPELINE = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.6,
        random_state=RANDOM_STATE,
    )
    """
).strip()


GB_READY_PLOT = dedent(
    """
    gb_importances = pd.Series(fitted_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)

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


READY_NOTEBOOK_SPECS = [
    {
        "filename": "Challenge_01_KNN_Ready.ipynb",
        "title": "# Challenge 01 Ready: K-nearest neighbors",
        "model_setup": KNN_READY_MODEL,
        "model_plot": KNN_READY_PLOT,
        "conclusion": KNN_READY_CONCLUSION,
    },
    {
        "filename": "Challenge_02_LogisticRegression_Ready.ipynb",
        "title": "# Challenge 02 Ready: Logistic regression",
        "model_setup": LOG_READY_MODEL,
        "model_plot": LOG_READY_PLOT,
        "conclusion": LOG_CONCLUSION,
    },
    {
        "filename": "Challenge_03_DecisionTree_Ready.ipynb",
        "title": "# Challenge 03 Ready: Decision tree",
        "model_setup": TREE_READY_MODEL,
        "model_plot": TREE_READY_PLOT,
        "conclusion": TREE_CONCLUSION,
    },
    {
        "filename": "Challenge_04_RandomForest_Ready.ipynb",
        "title": "# Challenge 04 Ready: Random forest",
        "model_setup": RF_READY_MODEL,
        "model_plot": RF_READY_PLOT,
        "conclusion": RF_CONCLUSION,
    },
    {
        "filename": "Challenge_05_GradientBoosting_Ready.ipynb",
        "title": "# Challenge 05 Ready: Gradient boosting",
        "model_setup": GB_READY_MODEL,
        "model_plot": GB_READY_PLOT,
        "conclusion": GB_CONCLUSION,
    },
]


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_ready_notebook(spec: dict) -> nbf.NotebookNode:
    cells = [
        markdown_cell(spec["title"]),
        markdown_cell(READY_INTRO),
        markdown_cell("## 0. Load libraries"),
        code_cell(COMMON_IMPORTS),
        markdown_cell("## 1. Configure the model with the fixed best hyperparameters"),
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
        markdown_cell("## 11. Fit the model with the fixed best hyperparameters"),
        code_cell(READY_FIT),
        markdown_cell("## 12. Evaluate the model on the validation split"),
        code_cell(READY_VALIDATION),
        markdown_cell("## 13. Interpret the fitted model"),
        code_cell(spec["model_plot"]),
        markdown_cell(spec["conclusion"]),
        markdown_cell("## 14. Refit on the full training set and generate submission"),
        code_cell(READY_FINAL_MODEL),
        markdown_cell("## 15. Save a compact experiment summary"),
        code_cell(READY_SUMMARY),
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
    for spec in READY_NOTEBOOK_SPECS:
        notebook = build_ready_notebook(spec)
        path = ROOT / spec["filename"]
        nbf.write(notebook, path)
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
