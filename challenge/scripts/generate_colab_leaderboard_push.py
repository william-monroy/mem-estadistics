from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "colab_leaderboard_push_ultra"
NOTEBOOK_PATH = TARGET_DIR / "Challenge_13_Leaderboard_Push_Colab.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Challenge 13: Public Leaderboard Push

        Esta notebook esta diseñada para el ultimo empuje sobre el leaderboard publico.

        Objetivo:

        1. reutilizar los mejores artefactos ya existentes;
        2. reentrenar solo candidatos elite y cercanos al frente de Pareto;
        3. construir un ensamble mas fino que el `challenge_12_final_stacking_colab`;
        4. opcionalmente generar submissions agresivos con pseudo-labeling de alta confianza.

        La idea no es volver a abrir una busqueda enorme. La idea es exprimir mejor tres fuentes de mejora:

        - diversidad entre variantes fuertes de `signal_features`;
        - una pequeña cuota de diversidad ortogonal via `raw SVM` y `KNN`;
        - reduccion de varianza via repeated CV y ensamble fino.
        """
    ),
    code(
        """
        import json
        import math
        import os
        import random
        import shutil
        import time
        import zipfile
        from dataclasses import dataclass
        from itertools import combinations
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from scipy.stats import rankdata
        from sklearn.base import clone
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
        from sklearn.svm import SVC

        try:
            from google.colab import files
            IN_COLAB = True
        except Exception:
            files = None
            IN_COLAB = False

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 200)
        np.set_printoptions(suppress=True, precision=6)
        """
    ),
    code(
        """
        # Configuration

        WORKSPACE_ROOT = Path("/content/challenge_public_lb_push_workspace")

        UPLOAD_DATA_FILES = False
        UPLOAD_RESULT_BUNDLES = False
        RESTORE_RUN_BUNDLE = False
        RUN_STAGE_1_DISCOVER_EXISTING = True
        RUN_STAGE_2_TRAIN_TARGETED_MODELS = True
        RUN_STAGE_3_ENSEMBLE_SEARCH = True
        RUN_STAGE_4_AGGRESSIVE_PSEUDOLABEL = True
        EXPORT_BUNDLE_AT_END = True

        SEARCH_PROFILE = "aggressive"  # "balanced" or "aggressive"
        USE_CACHED_RESULTS = True
        N_JOBS = -1
        RANDOM_STATE = 42

        DATA_FILES = {
            "training.csv": WORKSPACE_ROOT / "data" / "training.csv",
            "test.csv": WORKSPACE_ROOT / "data" / "test.csv",
            "sample.csv": WORKSPACE_ROOT / "data" / "sample.csv",
        }

        OUTPUT_ROOT = WORKSPACE_ROOT / "output" / "challenge_13_public_lb_push_colab"
        CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
        CANDIDATE_ROOT = OUTPUT_ROOT / "candidate_models"
        SUBMISSION_ROOT = WORKSPACE_ROOT / "submissions"
        EXPORT_ROOT = WORKSPACE_ROOT / "exports"
        UPLOAD_ROOT = WORKSPACE_ROOT / "uploads"
        EXTERNAL_ROOT = WORKSPACE_ROOT / "external_artifacts"

        SAFE_SUBMISSION_NAME = "challenge_13_public_lb_push_safe_submission.csv"
        AGGR15_SUBMISSION_NAME = "challenge_13_public_lb_push_aggressive_alpha15.csv"
        AGGR25_SUBMISSION_NAME = "challenge_13_public_lb_push_aggressive_alpha25.csv"
        PURE_PSEUDO_SUBMISSION_NAME = "challenge_13_public_lb_push_pseudo_only.csv"
        """
    ),
    code(
        """
        def ensure_dirs() -> None:
            for path in [
                WORKSPACE_ROOT,
                WORKSPACE_ROOT / "data",
                OUTPUT_ROOT,
                CHECKPOINT_ROOT,
                CANDIDATE_ROOT,
                SUBMISSION_ROOT,
                EXPORT_ROOT,
                UPLOAD_ROOT,
                EXTERNAL_ROOT,
            ]:
                path.mkdir(parents=True, exist_ok=True)


        def save_json(path: Path, payload: dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)


        def set_seed(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)


        ensure_dirs()
        set_seed(RANDOM_STATE)
        print("Workspace:", WORKSPACE_ROOT)
        """
    ),
    md(
        """
        ## Optional: upload data files

        Si estas iniciando una sesion nueva de Colab, activa `UPLOAD_DATA_FILES = True` y sube:

        - `training.csv`
        - `test.csv`
        - `sample.csv`
        """
    ),
    code(
        """
        if UPLOAD_DATA_FILES:
            assert IN_COLAB, "This upload cell is intended for Google Colab."
            uploaded = files.upload()
            for name, content in uploaded.items():
                if name in DATA_FILES:
                    DATA_FILES[name].write_bytes(content)
                    print("Saved", name, "to", DATA_FILES[name])

        missing = [name for name, path in DATA_FILES.items() if not path.exists()]
        if missing:
            print("Missing data files:", missing)
        else:
            print("All required data files are present.")
        """
    ),
    md(
        """
        ## Optional: upload current result bundles

        Esta notebook puede reutilizar artefactos ya generados. Lo mas util es subir el bundle preparado en la carpeta local:

        - `current_artifacts_for_lb_push.zip`

        Tambien puedes subir ZIPs adicionales si quieres usar otras corridas.
        """
    ),
    code(
        """
        if UPLOAD_RESULT_BUNDLES:
            assert IN_COLAB, "This upload cell is intended for Google Colab."
            uploaded = files.upload()
            for name, content in uploaded.items():
                bundle_path = UPLOAD_ROOT / name
                bundle_path.write_bytes(content)
                print("Uploaded bundle:", bundle_path)

        for bundle_path in sorted(UPLOAD_ROOT.glob("*.zip")):
            extract_root = EXTERNAL_ROOT / bundle_path.stem
            if extract_root.exists():
                print("Bundle already extracted:", bundle_path.name)
                continue
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle_path, "r") as zf:
                zf.extractall(extract_root)
            print("Extracted:", bundle_path.name, "->", extract_root)
        """
    ),
    code(
        """
        train_df = pd.read_csv(DATA_FILES["training.csv"])
        test_df = pd.read_csv(DATA_FILES["test.csv"])
        sample_df = pd.read_csv(DATA_FILES["sample.csv"])

        feature_cols = [c for c in train_df.columns if c.startswith("V")]
        X_all = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_all = train_df["class"].to_numpy(dtype=np.int8)
        X_test_all = test_df[feature_cols].to_numpy(dtype=np.float32)
        test_ids = test_df["id"].to_numpy(dtype=np.int64)
        train_ids = train_df["id"].to_numpy(dtype=np.int64)

        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
        print("Feature count:", len(feature_cols))
        print("Class balance:", train_df["class"].value_counts().to_dict())
        """
    ),
    code(
        """
        def build_feature_bank(X_input: np.ndarray) -> dict[str, np.ndarray]:
            X = np.asarray(X_input, dtype=np.float64)
            mean = X.mean(axis=1)
            std = X.std(axis=1)
            median = np.median(X, axis=1)
            q10 = np.quantile(X, 0.10, axis=1)
            q25 = np.quantile(X, 0.25, axis=1)
            q75 = np.quantile(X, 0.75, axis=1)
            q90 = np.quantile(X, 0.90, axis=1)
            rms = np.sqrt(np.mean(X ** 2, axis=1))
            energy = np.sum(X ** 2, axis=1)
            peak = np.max(np.abs(X), axis=1)
            peak_to_peak = X.max(axis=1) - X.min(axis=1)
            centered = X - mean[:, None]
            skew = np.mean(centered ** 3, axis=1) / ((std ** 3) + 1e-12)
            kurt = np.mean(centered ** 4, axis=1) / ((std ** 4) + 1e-12)
            zcr = ((X[:, 1:] * X[:, :-1]) < 0).mean(axis=1)

            basic = np.column_stack(
                [
                    mean,
                    std,
                    median,
                    q10,
                    q25,
                    q75,
                    q90,
                    rms,
                    energy,
                    peak,
                    peak_to_peak,
                    skew,
                    kurt,
                    zcr,
                ]
            )

            fft_mag = np.abs(np.fft.rfft(X, axis=1))
            fft_power = fft_mag ** 2
            freqs = np.arange(fft_power.shape[1], dtype=np.float64)
            power_sum = fft_power.sum(axis=1) + 1e-12
            spectral_centroid = (fft_power * freqs[None, :]).sum(axis=1) / power_sum
            dominant_freq = np.argmax(fft_power, axis=1)
            normalized_power = fft_power / power_sum[:, None]
            spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-12), axis=1)

            band_slices = np.array_split(np.arange(fft_power.shape[1]), 6)
            band_energy = [fft_power[:, idx].sum(axis=1) for idx in band_slices]
            fft_features = np.column_stack([spectral_centroid, dominant_freq, spectral_entropy, *band_energy])

            segments = np.array_split(np.arange(X.shape[1]), 4)
            segment_features = []
            for idx in segments:
                segment = X[:, idx]
                segment_features.extend(
                    [
                        segment.mean(axis=1),
                        segment.std(axis=1),
                        np.sqrt(np.mean(segment ** 2, axis=1)),
                        segment.max(axis=1) - segment.min(axis=1),
                    ]
                )
            segment_features = np.column_stack(segment_features)

            return {
                "basic": basic.astype(np.float32),
                "fft": np.hstack([basic, fft_features]).astype(np.float32),
                "all": np.hstack([basic, fft_features, segment_features]).astype(np.float32),
            }


        def fit_scaler(name: str):
            if name == "standard":
                return StandardScaler()
            if name == "robust":
                return RobustScaler()
            if name == "power_yeo":
                return PowerTransformer(method="yeo-johnson", standardize=True)
            if name == "quantile_normal":
                return QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
            raise ValueError(f"Unknown scale method: {name}")


        def apply_lof_by_class(X_input: np.ndarray, y_input: np.ndarray, contamination: float) -> tuple[np.ndarray, np.ndarray]:
            scaled = StandardScaler().fit_transform(X_input)
            keep_mask = np.ones(len(X_input), dtype=bool)
            for cls in np.unique(y_input):
                idx = np.where(y_input == cls)[0]
                if len(idx) < 20:
                    continue
                n_neighbors = min(25, len(idx) - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                pred = lof.fit_predict(scaled[idx])
                keep_mask[idx] = pred == 1
            return X_input[keep_mask], y_input[keep_mask]


        def build_signal_train_eval_test(X_fit_raw, X_eval_raw, X_test_raw, params):
            bank_fit = build_feature_bank(X_fit_raw)
            bank_eval = build_feature_bank(X_eval_raw)
            bank_test = build_feature_bank(X_test_raw)

            X_fit = bank_fit[params["feature__set"]]
            X_eval = bank_eval[params["feature__set"]]
            X_test = bank_test[params["feature__set"]]

            raw_pca_dim = int(params.get("feature__raw_pca", 0))
            if raw_pca_dim > 0:
                raw_scaler = StandardScaler()
                X_fit_scaled = raw_scaler.fit_transform(X_fit_raw)
                X_eval_scaled = raw_scaler.transform(X_eval_raw)
                X_test_scaled = raw_scaler.transform(X_test_raw)
                pca = PCA(n_components=min(raw_pca_dim, X_fit_scaled.shape[0], X_fit_scaled.shape[1]), random_state=RANDOM_STATE)
                X_fit_raw_pca = pca.fit_transform(X_fit_scaled).astype(np.float32)
                X_eval_raw_pca = pca.transform(X_eval_scaled).astype(np.float32)
                X_test_raw_pca = pca.transform(X_test_scaled).astype(np.float32)
                X_fit = np.hstack([X_fit, X_fit_raw_pca]).astype(np.float32)
                X_eval = np.hstack([X_eval, X_eval_raw_pca]).astype(np.float32)
                X_test = np.hstack([X_test, X_test_raw_pca]).astype(np.float32)

            scaler = fit_scaler(params["scale__method"])
            X_fit = scaler.fit_transform(X_fit).astype(np.float32)
            X_eval = scaler.transform(X_eval).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)
            return X_fit, X_eval, X_test


        def build_raw_svm_train_eval_test(X_fit_raw, y_fit, X_eval_raw, X_test_raw, params):
            clean_method = params["clean__method"]
            X_model = X_fit_raw
            y_model = y_fit
            removed_rows = 0
            if clean_method.startswith("lof_"):
                contamination = float(clean_method.split("_")[1])
                before = len(X_model)
                X_model, y_model = apply_lof_by_class(X_model, y_model, contamination)
                removed_rows = before - len(X_model)

            scaler = fit_scaler(params["scale__method"])
            X_model_scaled = scaler.fit_transform(X_model).astype(np.float32)
            X_eval_scaled = scaler.transform(X_eval_raw).astype(np.float32)
            X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

            pca = PCA(
                n_components=min(int(params["pca__n_components"]), X_model_scaled.shape[0], X_model_scaled.shape[1]),
                random_state=RANDOM_STATE,
            )
            X_model_pca = pca.fit_transform(X_model_scaled).astype(np.float32)
            X_eval_pca = pca.transform(X_eval_scaled).astype(np.float32)
            X_test_pca = pca.transform(X_test_scaled).astype(np.float32)
            return X_model_pca, y_model, X_eval_pca, X_test_pca, removed_rows


        def build_knn_train_eval_test(X_fit_raw, y_fit, X_eval_raw, X_test_raw, params):
            clean_method = params["clean__method"]
            X_model = X_fit_raw
            y_model = y_fit
            removed_rows = 0
            if clean_method.startswith("lof_"):
                contamination = float(clean_method.split("_")[1])
                before = len(X_model)
                X_model, y_model = apply_lof_by_class(X_model, y_model, contamination)
                removed_rows = before - len(X_model)

            scaler = fit_scaler(params["scale__method"])
            X_model_scaled = scaler.fit_transform(X_model).astype(np.float32)
            X_eval_scaled = scaler.transform(X_eval_raw).astype(np.float32)
            X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

            pca = PCA(
                n_components=min(int(params["pca__n_components"]), X_model_scaled.shape[0], X_model_scaled.shape[1]),
                random_state=RANDOM_STATE,
            )
            X_model_pca = pca.fit_transform(X_model_scaled).astype(np.float32)
            X_eval_pca = pca.transform(X_eval_scaled).astype(np.float32)
            X_test_pca = pca.transform(X_test_scaled).astype(np.float32)
            return X_model_pca, y_model, X_eval_pca, X_test_pca, removed_rows
        """
    ),
    code(
        """
        def make_signal_model(params):
            return SVC(
                C=float(params["model__C"]),
                gamma=float(params["model__gamma"]),
                kernel="rbf",
                probability=True,
                cache_size=512,
                random_state=RANDOM_STATE,
            )


        def make_raw_svm_model(params):
            return SVC(
                C=float(params["model__C"]),
                gamma=float(params["model__gamma"]),
                kernel="rbf",
                probability=True,
                cache_size=512,
                random_state=RANDOM_STATE,
            )


        def make_knn_model(params):
            return KNeighborsClassifier(
                n_neighbors=int(params["model__n_neighbors"]),
                metric=params["model__metric"],
                weights=params.get("model__weights", "distance"),
                n_jobs=N_JOBS,
            )


        def save_probability_artifacts(model_name: str, oof_prob: np.ndarray, test_prob: np.ndarray, y_true: np.ndarray) -> dict:
            model_dir = CANDIDATE_ROOT / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            oof_df = pd.DataFrame(
                {
                    "id": train_ids,
                    "y_true": y_true.astype(np.int8),
                    "prob_1": oof_prob.astype(np.float32),
                    "pred": (oof_prob >= 0.5).astype(np.int8),
                    "source_model": model_name,
                }
            )
            test_df_prob = pd.DataFrame(
                {
                    "id": test_ids,
                    "prob_1": test_prob.astype(np.float32),
                    "pred": (test_prob >= 0.5).astype(np.int8),
                    "source_model": model_name,
                }
            )
            oof_path = model_dir / "oof_probabilities.csv"
            test_path = model_dir / "test_probabilities.csv"
            oof_df.to_csv(oof_path, index=False)
            test_df_prob.to_csv(test_path, index=False)
            return {"oof_path": str(oof_path), "test_path": str(test_path)}


        def train_signal_candidate(model_name: str, params: dict, seeds: list[int]) -> dict:
            model_dir = CANDIDATE_ROOT / model_name
            summary_path = model_dir / "summary.json"
            if USE_CACHED_RESULTS and summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            n_samples = len(X_all)
            oof_sum = np.zeros(n_samples, dtype=np.float64)
            oof_count = np.zeros(n_samples, dtype=np.float64)
            test_sum = np.zeros(len(X_test_all), dtype=np.float64)
            fold_rows = []

            for seed in seeds:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold_idx, (fit_idx, eval_idx) in enumerate(cv.split(X_all, y_all)):
                    start = time.time()
                    X_fit, X_eval, X_test = build_signal_train_eval_test(
                        X_all[fit_idx], X_all[eval_idx], X_test_all, params
                    )
                    model = make_signal_model(params)
                    model.fit(X_fit, y_all[fit_idx])
                    eval_prob = model.predict_proba(X_eval)[:, 1]
                    test_prob = model.predict_proba(X_test)[:, 1]
                    elapsed = time.time() - start

                    oof_sum[eval_idx] += eval_prob
                    oof_count[eval_idx] += 1.0
                    test_sum += test_prob
                    fold_rows.append(
                        {
                            "model_name": model_name,
                            "seed": seed,
                            "fold_idx": fold_idx,
                            "fold_accuracy": accuracy_score(y_all[eval_idx], (eval_prob >= 0.5).astype(np.int8)),
                            "fit_seconds": elapsed,
                        }
                    )

            oof_prob = (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32)
            test_prob = (test_sum / (len(seeds) * 5)).astype(np.float32)
            saved = save_probability_artifacts(model_name, oof_prob, test_prob, y_all)

            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(model_dir / "cv_fold_results.csv", index=False)
            summary = {
                "model_name": model_name,
                "family": "signal_features_svm",
                "params": params,
                "seeds": seeds,
                "oof_accuracy": float(accuracy_score(y_all, (oof_prob >= 0.5).astype(np.int8))),
                "oof_confusion_matrix": confusion_matrix(y_all, (oof_prob >= 0.5).astype(np.int8)).tolist(),
                "mean_fit_seconds": float(fold_df["fit_seconds"].mean()),
                "oof_path": saved["oof_path"],
                "test_path": saved["test_path"],
            }
            save_json(summary_path, summary)
            return summary


        def train_raw_svm_candidate(model_name: str, params: dict, seeds: list[int]) -> dict:
            model_dir = CANDIDATE_ROOT / model_name
            summary_path = model_dir / "summary.json"
            if USE_CACHED_RESULTS and summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            n_samples = len(X_all)
            oof_sum = np.zeros(n_samples, dtype=np.float64)
            oof_count = np.zeros(n_samples, dtype=np.float64)
            test_sum = np.zeros(len(X_test_all), dtype=np.float64)
            fold_rows = []

            for seed in seeds:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold_idx, (fit_idx, eval_idx) in enumerate(cv.split(X_all, y_all)):
                    start = time.time()
                    X_fit_model, y_fit_model, X_eval_model, X_test_model, removed_rows = build_raw_svm_train_eval_test(
                        X_all[fit_idx], y_all[fit_idx], X_all[eval_idx], X_test_all, params
                    )
                    model = make_raw_svm_model(params)
                    model.fit(X_fit_model, y_fit_model)
                    eval_prob = model.predict_proba(X_eval_model)[:, 1]
                    test_prob = model.predict_proba(X_test_model)[:, 1]
                    elapsed = time.time() - start

                    oof_sum[eval_idx] += eval_prob
                    oof_count[eval_idx] += 1.0
                    test_sum += test_prob
                    fold_rows.append(
                        {
                            "model_name": model_name,
                            "seed": seed,
                            "fold_idx": fold_idx,
                            "fold_accuracy": accuracy_score(y_all[eval_idx], (eval_prob >= 0.5).astype(np.int8)),
                            "fit_seconds": elapsed,
                            "removed_rows": removed_rows,
                        }
                    )

            oof_prob = (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32)
            test_prob = (test_sum / (len(seeds) * 5)).astype(np.float32)
            saved = save_probability_artifacts(model_name, oof_prob, test_prob, y_all)

            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(model_dir / "cv_fold_results.csv", index=False)
            summary = {
                "model_name": model_name,
                "family": "raw_svm_diversity",
                "params": params,
                "seeds": seeds,
                "oof_accuracy": float(accuracy_score(y_all, (oof_prob >= 0.5).astype(np.int8))),
                "oof_confusion_matrix": confusion_matrix(y_all, (oof_prob >= 0.5).astype(np.int8)).tolist(),
                "mean_fit_seconds": float(fold_df["fit_seconds"].mean()),
                "mean_removed_rows": float(fold_df["removed_rows"].mean()),
                "oof_path": saved["oof_path"],
                "test_path": saved["test_path"],
            }
            save_json(summary_path, summary)
            return summary


        def train_knn_candidate(model_name: str, params: dict, seeds: list[int]) -> dict:
            model_dir = CANDIDATE_ROOT / model_name
            summary_path = model_dir / "summary.json"
            if USE_CACHED_RESULTS and summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            n_samples = len(X_all)
            oof_sum = np.zeros(n_samples, dtype=np.float64)
            oof_count = np.zeros(n_samples, dtype=np.float64)
            test_sum = np.zeros(len(X_test_all), dtype=np.float64)
            fold_rows = []

            for seed in seeds:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold_idx, (fit_idx, eval_idx) in enumerate(cv.split(X_all, y_all)):
                    start = time.time()
                    X_fit_model, y_fit_model, X_eval_model, X_test_model, removed_rows = build_knn_train_eval_test(
                        X_all[fit_idx], y_all[fit_idx], X_all[eval_idx], X_test_all, params
                    )
                    model = make_knn_model(params)
                    model.fit(X_fit_model, y_fit_model)
                    eval_prob = model.predict_proba(X_eval_model)[:, 1]
                    test_prob = model.predict_proba(X_test_model)[:, 1]
                    elapsed = time.time() - start

                    oof_sum[eval_idx] += eval_prob
                    oof_count[eval_idx] += 1.0
                    test_sum += test_prob
                    fold_rows.append(
                        {
                            "model_name": model_name,
                            "seed": seed,
                            "fold_idx": fold_idx,
                            "fold_accuracy": accuracy_score(y_all[eval_idx], (eval_prob >= 0.5).astype(np.int8)),
                            "fit_seconds": elapsed,
                            "removed_rows": removed_rows,
                        }
                    )

            oof_prob = (oof_sum / np.maximum(oof_count, 1.0)).astype(np.float32)
            test_prob = (test_sum / (len(seeds) * 5)).astype(np.float32)
            saved = save_probability_artifacts(model_name, oof_prob, test_prob, y_all)

            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(model_dir / "cv_fold_results.csv", index=False)
            summary = {
                "model_name": model_name,
                "family": "knn_diversity",
                "params": params,
                "seeds": seeds,
                "oof_accuracy": float(accuracy_score(y_all, (oof_prob >= 0.5).astype(np.int8))),
                "oof_confusion_matrix": confusion_matrix(y_all, (oof_prob >= 0.5).astype(np.int8)).tolist(),
                "mean_fit_seconds": float(fold_df["fit_seconds"].mean()),
                "mean_removed_rows": float(fold_df["removed_rows"].mean()),
                "oof_path": saved["oof_path"],
                "test_path": saved["test_path"],
            }
            save_json(summary_path, summary)
            return summary
        """
    ),
    code(
        """
        SIGNAL_CANDIDATES_BALANCED = {
            "signal_all_raw32_std_c6_g001": {
                "feature__set": "all",
                "feature__raw_pca": 32,
                "scale__method": "standard",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
            "signal_all_raw32_rob_c6_g001": {
                "feature__set": "all",
                "feature__raw_pca": 32,
                "scale__method": "robust",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
            "signal_all_raw16_std_c6_g001": {
                "feature__set": "all",
                "feature__raw_pca": 16,
                "scale__method": "standard",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
            "signal_fft_raw32_std_c6_g001": {
                "feature__set": "fft",
                "feature__raw_pca": 32,
                "scale__method": "standard",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
        }

        SIGNAL_CANDIDATES_AGGRESSIVE = {
            **SIGNAL_CANDIDATES_BALANCED,
            "signal_all_raw32_std_c3_g001": {
                "feature__set": "all",
                "feature__raw_pca": 32,
                "scale__method": "standard",
                "model__family": "svm",
                "model__C": 3.0,
                "model__gamma": 0.01,
            },
            "signal_fft_raw32_rob_c6_g001": {
                "feature__set": "fft",
                "feature__raw_pca": 32,
                "scale__method": "robust",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
            "signal_all_raw0_rob_c6_g001": {
                "feature__set": "all",
                "feature__raw_pca": 0,
                "scale__method": "robust",
                "model__family": "svm",
                "model__C": 6.0,
                "model__gamma": 0.01,
            },
        }

        RAW_SVM_CANDIDATES_BALANCED = {
            "raw_svm_lof03_rob_pca120_c4_g002": {
                "clean__method": "lof_0.03",
                "scale__method": "robust",
                "pca__n_components": 120,
                "model__C": 4.0,
                "model__gamma": 0.02,
            }
        }

        RAW_SVM_CANDIDATES_AGGRESSIVE = {
            **RAW_SVM_CANDIDATES_BALANCED,
            "raw_svm_lof03_power_pca136_c3_g001": {
                "clean__method": "lof_0.03",
                "scale__method": "power_yeo",
                "pca__n_components": 136,
                "model__C": 3.0,
                "model__gamma": 0.01,
            },
        }

        KNN_CANDIDATES = {
            "knn_clean_best_refresh": {
                "clean__method": "none",
                "scale__method": "standard",
                "pca__n_components": 64,
                "model__n_neighbors": 2,
                "model__metric": "manhattan",
                "model__weights": "distance",
            }
        }

        if SEARCH_PROFILE == "balanced":
            REPEATED_SEEDS = [13, 29]
            SIGNAL_CANDIDATES = SIGNAL_CANDIDATES_BALANCED
            RAW_SVM_CANDIDATES = RAW_SVM_CANDIDATES_BALANCED
            RANDOM_WEIGHT_SAMPLES = 100
        else:
            REPEATED_SEEDS = [13, 29, 47]
            SIGNAL_CANDIDATES = SIGNAL_CANDIDATES_AGGRESSIVE
            RAW_SVM_CANDIDATES = RAW_SVM_CANDIDATES_AGGRESSIVE
            RANDOM_WEIGHT_SAMPLES = 240

        print("Signal candidates:", list(SIGNAL_CANDIDATES))
        print("Raw SVM candidates:", list(RAW_SVM_CANDIDATES))
        print("KNN candidates:", list(KNN_CANDIDATES))
        print("Repeated seeds:", REPEATED_SEEDS)
        """
    ),
    code(
        """
        def discover_existing_artifacts() -> pd.DataFrame:
            rows = []
            for search_root in [EXTERNAL_ROOT, CANDIDATE_ROOT]:
                if not search_root.exists():
                    continue
                for oof_name, test_name in [
                    ("oof_probabilities.csv", "test_probabilities.csv"),
                    ("meta_oof_probabilities.csv", "meta_test_probabilities.csv"),
                ]:
                    for oof_path in search_root.rglob(oof_name):
                        test_path = oof_path.with_name(test_name)
                        if not test_path.exists():
                            continue
                        try:
                            oof_df = pd.read_csv(oof_path)
                            test_df_prob = pd.read_csv(test_path)
                        except Exception:
                            continue
                        model_name = str(oof_df.get("source_model", pd.Series([oof_path.parent.name])).iloc[0])
                        if "prob_1" not in oof_df.columns or "prob_1" not in test_df_prob.columns:
                            continue
                        if "y_true" not in oof_df.columns:
                            continue
                        oof_acc = accuracy_score(oof_df["y_true"], (oof_df["prob_1"] >= 0.5).astype(np.int8))
                        rows.append(
                            {
                                "model_name": model_name,
                                "oof_accuracy": float(oof_acc),
                                "oof_path": str(oof_path),
                                "test_path": str(test_path),
                                "search_root": str(search_root),
                            }
                        )
            if not rows:
                return pd.DataFrame(columns=["model_name", "oof_accuracy", "oof_path", "test_path", "search_root"])
            df = pd.DataFrame(rows).drop_duplicates(subset=["model_name", "oof_path"]).sort_values(["oof_accuracy", "model_name"], ascending=[False, True])
            df.to_csv(CHECKPOINT_ROOT / "discovered_artifacts.csv", index=False)
            return df


        discovered_df = discover_existing_artifacts() if RUN_STAGE_1_DISCOVER_EXISTING else pd.DataFrame()
        display(discovered_df.head(20))
        """
    ),
    code(
        """
        training_summaries = []

        if RUN_STAGE_2_TRAIN_TARGETED_MODELS:
            for model_name, params in SIGNAL_CANDIDATES.items():
                print("Training signal candidate:", model_name)
                training_summaries.append(train_signal_candidate(model_name, params, REPEATED_SEEDS))

            for model_name, params in RAW_SVM_CANDIDATES.items():
                print("Training raw SVM candidate:", model_name)
                training_summaries.append(train_raw_svm_candidate(model_name, params, REPEATED_SEEDS))

            if "challenge_08_knn_cleaning_colab_ultra" not in discovered_df.get("model_name", pd.Series(dtype=object)).tolist():
                for model_name, params in KNN_CANDIDATES.items():
                    print("Training KNN candidate:", model_name)
                    training_summaries.append(train_knn_candidate(model_name, params, REPEATED_SEEDS))

        if training_summaries:
            pd.DataFrame(training_summaries).to_csv(CHECKPOINT_ROOT / "trained_candidate_summaries.csv", index=False)

        artifact_registry = discover_existing_artifacts()
        display(artifact_registry.head(50))
        """
    ),
    code(
        """
        def load_probability_pair(oof_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            oof_df = pd.read_csv(oof_path)
            test_df_prob = pd.read_csv(test_path)
            return oof_df, test_df_prob


        def deduplicate_models(registry: pd.DataFrame) -> pd.DataFrame:
            if registry.empty:
                return registry
            keep_rows = []
            kept_probs = []
            for _, row in registry.sort_values("oof_accuracy", ascending=False).iterrows():
                oof_df, _ = load_probability_pair(row["oof_path"], row["test_path"])
                prob = oof_df["prob_1"].to_numpy(dtype=np.float64)
                is_dup = False
                for prev in kept_probs:
                    corr = np.corrcoef(prob, prev)[0, 1]
                    if np.isfinite(corr) and corr > 0.9995:
                        is_dup = True
                        break
                if not is_dup:
                    keep_rows.append(row.to_dict())
                    kept_probs.append(prob)
            return pd.DataFrame(keep_rows)


        def build_meta_frames(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            oof_base = pd.DataFrame({"id": train_ids, "y_true": y_all})
            test_base = pd.DataFrame({"id": test_ids})
            for _, row in registry.iterrows():
                oof_df, test_df_prob = load_probability_pair(row["oof_path"], row["test_path"])
                model_name = row["model_name"]
                oof_base = oof_base.merge(oof_df[["id", "prob_1"]].rename(columns={"prob_1": model_name}), on="id", how="left")
                test_base = test_base.merge(test_df_prob[["id", "prob_1"]].rename(columns={"prob_1": model_name}), on="id", how="left")
            return oof_base, test_base


        def threshold_search(y_true: np.ndarray, prob: np.ndarray, threshold_grid: np.ndarray) -> tuple[float, float]:
            best_t = 0.5
            best_acc = -1.0
            for t in threshold_grid:
                acc = accuracy_score(y_true, (prob >= t).astype(np.int8))
                if acc > best_acc:
                    best_acc = acc
                    best_t = float(t)
            return best_t, float(best_acc)


        def random_weight_matrix(n_models: int, n_samples: int) -> np.ndarray:
            alpha = np.ones(n_models, dtype=np.float64)
            weights = np.random.default_rng(RANDOM_STATE + n_models + n_samples).dirichlet(alpha, size=n_samples)
            return weights


        def weighted_prob(df: pd.DataFrame, cols: list[str], weights: np.ndarray) -> np.ndarray:
            return df[cols].to_numpy(dtype=np.float64) @ weights.astype(np.float64)


        def rank_average_prob(df: pd.DataFrame, cols: list[str], weights: np.ndarray | None = None) -> np.ndarray:
            matrix = df[cols].to_numpy(dtype=np.float64)
            ranked = np.vstack([rankdata(matrix[:, idx], method="average") / len(df) for idx in range(matrix.shape[1])]).T
            if weights is None:
                weights = np.ones(matrix.shape[1], dtype=np.float64) / matrix.shape[1]
            return ranked @ weights.astype(np.float64)


        def generate_logreg_oof(meta_df: pd.DataFrame, cols: list[str], c_value: float) -> tuple[np.ndarray, np.ndarray]:
            X_meta = meta_df[cols].to_numpy(dtype=np.float32)
            y_meta = meta_df["y_true"].to_numpy(dtype=np.int8)
            X_test_meta = meta_test_df[cols].to_numpy(dtype=np.float32)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            oof_prob = np.zeros(len(meta_df), dtype=np.float32)
            test_prob_sum = np.zeros(len(meta_test_df), dtype=np.float32)
            for fit_idx, eval_idx in cv.split(X_meta, y_meta):
                model = LogisticRegression(C=float(c_value), max_iter=2500)
                model.fit(X_meta[fit_idx], y_meta[fit_idx])
                oof_prob[eval_idx] = model.predict_proba(X_meta[eval_idx])[:, 1].astype(np.float32)
                test_prob_sum += model.predict_proba(X_test_meta)[:, 1].astype(np.float32)
            return oof_prob, (test_prob_sum / 5.0).astype(np.float32)
        """
    ),
    code(
        """
        ensemble_summary = None
        ensemble_search_df = pd.DataFrame()

        if RUN_STAGE_3_ENSEMBLE_SEARCH:
            registry = discover_existing_artifacts()
            if registry.empty:
                raise RuntimeError("No probability artifacts found. Upload bundles or train targeted models first.")

            registry = deduplicate_models(registry)
            registry = registry.sort_values("oof_accuracy", ascending=False).reset_index(drop=True)
            registry.to_csv(CHECKPOINT_ROOT / "ensemble_registry_deduplicated.csv", index=False)

            top_limit = 8 if SEARCH_PROFILE == "aggressive" else 6
            top_registry = registry.head(top_limit).copy()
            print("Models entering ensemble search:")
            display(top_registry[["model_name", "oof_accuracy"]])

            meta_df, meta_test_df = build_meta_frames(top_registry)
            meta_df.to_csv(CHECKPOINT_ROOT / "ensemble_meta_oof_input.csv", index=False)
            meta_test_df.to_csv(CHECKPOINT_ROOT / "ensemble_meta_test_input.csv", index=False)

            available_models = [c for c in meta_df.columns if c not in {"id", "y_true"}]
            threshold_grid = np.linspace(0.35, 0.65, 61)
            rows = []
            candidate_counter = 0

            for subset_size in range(2, min(5, len(available_models) + 1)):
                for subset in combinations(available_models, subset_size):
                    subset = list(subset)
                    if subset_size == 2:
                        for w0 in np.linspace(0.0, 1.0, 41):
                            weights = np.array([w0, 1.0 - w0], dtype=np.float64)
                            prob = weighted_prob(meta_df, subset, weights)
                            threshold, acc = threshold_search(meta_df["y_true"].to_numpy(dtype=np.int8), prob, threshold_grid)
                            candidate_counter += 1
                            rows.append(
                                {
                                    "candidate_id": f"blend_{candidate_counter:04d}",
                                    "meta_family": "weighted_average",
                                    "base_models_json": json.dumps(subset),
                                    "weights_json": json.dumps(weights.tolist()),
                                    "C": None,
                                    "best_threshold": threshold,
                                    "meta_oof_accuracy": acc,
                                }
                            )
                        rank_prob = rank_average_prob(meta_df, subset)
                        threshold, acc = threshold_search(meta_df["y_true"].to_numpy(dtype=np.int8), rank_prob, threshold_grid)
                        candidate_counter += 1
                        rows.append(
                            {
                                "candidate_id": f"blend_{candidate_counter:04d}",
                                "meta_family": "rank_average",
                                "base_models_json": json.dumps(subset),
                                "weights_json": None,
                                "C": None,
                                "best_threshold": threshold,
                                "meta_oof_accuracy": acc,
                            }
                        )
                        for c_value in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
                            logreg_oof, _ = generate_logreg_oof(meta_df, subset, c_value)
                            threshold, acc = threshold_search(meta_df["y_true"].to_numpy(dtype=np.int8), logreg_oof, threshold_grid)
                            candidate_counter += 1
                            rows.append(
                                {
                                    "candidate_id": f"blend_{candidate_counter:04d}",
                                    "meta_family": "logreg",
                                    "base_models_json": json.dumps(subset),
                                    "weights_json": None,
                                    "C": float(c_value),
                                    "best_threshold": threshold,
                                    "meta_oof_accuracy": acc,
                                }
                            )
                    else:
                        random_weights = random_weight_matrix(subset_size, RANDOM_WEIGHT_SAMPLES)
                        for weights in random_weights:
                            prob = weighted_prob(meta_df, subset, weights)
                            threshold, acc = threshold_search(meta_df["y_true"].to_numpy(dtype=np.int8), prob, threshold_grid)
                            candidate_counter += 1
                            rows.append(
                                {
                                    "candidate_id": f"blend_{candidate_counter:04d}",
                                    "meta_family": "weighted_average",
                                    "base_models_json": json.dumps(subset),
                                    "weights_json": json.dumps(np.round(weights, 6).tolist()),
                                    "C": None,
                                    "best_threshold": threshold,
                                    "meta_oof_accuracy": acc,
                                }
                            )

            ensemble_search_df = pd.DataFrame(rows).sort_values("meta_oof_accuracy", ascending=False).reset_index(drop=True)
            ensemble_search_df.to_csv(CHECKPOINT_ROOT / "leaderboard_push_search_results.csv", index=False)
            display(ensemble_search_df.head(20))

            best_row = ensemble_search_df.iloc[0].to_dict()
            best_models = json.loads(best_row["base_models_json"])
            y_true = meta_df["y_true"].to_numpy(dtype=np.int8)

            if best_row["meta_family"] == "weighted_average":
                weights = np.array(json.loads(best_row["weights_json"]), dtype=np.float64)
                best_oof_prob = weighted_prob(meta_df, best_models, weights).astype(np.float32)
                best_test_prob = weighted_prob(meta_test_df, best_models, weights).astype(np.float32)
            elif best_row["meta_family"] == "rank_average":
                best_oof_prob = rank_average_prob(meta_df, best_models).astype(np.float32)
                best_test_prob = rank_average_prob(meta_test_df, best_models).astype(np.float32)
            else:
                best_oof_prob, best_test_prob = generate_logreg_oof(meta_df, best_models, float(best_row["C"]))

            best_threshold = float(best_row["best_threshold"])
            safe_pred = (best_test_prob >= best_threshold).astype(np.int8)
            safe_submission = sample_df.copy()
            safe_submission["class"] = safe_pred
            safe_submission_path = SUBMISSION_ROOT / SAFE_SUBMISSION_NAME
            safe_submission.to_csv(safe_submission_path, index=False)

            meta_oof_df = pd.DataFrame(
                {
                    "id": train_ids,
                    "y_true": y_true,
                    "prob_1": best_oof_prob,
                    "pred": (best_oof_prob >= best_threshold).astype(np.int8),
                    "source_model": "challenge_13_public_lb_push_safe",
                }
            )
            meta_test_df_out = pd.DataFrame(
                {
                    "id": test_ids,
                    "prob_1": best_test_prob,
                    "pred": safe_pred,
                    "source_model": "challenge_13_public_lb_push_safe",
                }
            )
            meta_oof_df.to_csv(OUTPUT_ROOT / "meta_oof_probabilities.csv", index=False)
            meta_test_df_out.to_csv(OUTPUT_ROOT / "meta_test_probabilities.csv", index=False)

            ensemble_summary = {
                "model_name": "Public leaderboard push ensemble",
                "strategy": "elite_targeted_ensemble",
                "search_profile": SEARCH_PROFILE,
                "selected_base_models": best_models,
                "best_meta_family": best_row["meta_family"],
                "best_threshold": best_threshold,
                "best_meta_oof_accuracy": float(best_row["meta_oof_accuracy"]),
                "best_weights": None if pd.isna(best_row["weights_json"]) or best_row["weights_json"] is None else json.loads(best_row["weights_json"]),
                "best_C": None if pd.isna(best_row["C"]) else float(best_row["C"]),
                "safe_submission_path": str(safe_submission_path),
                "workspace_root": str(WORKSPACE_ROOT),
                "persist_root": str(OUTPUT_ROOT),
            }
            save_json(OUTPUT_ROOT / "summary.json", ensemble_summary)
            print("Safe submission:", safe_submission_path)
            print("Best meta OOF accuracy:", ensemble_summary["best_meta_oof_accuracy"])
        """
    ),
    code(
        """
        if RUN_STAGE_4_AGGRESSIVE_PSEUDOLABEL:
            if ensemble_summary is None:
                raise RuntimeError("Run Stage 3 first; pseudo-labeling depends on the best robust ensemble.")

            registry = pd.read_csv(CHECKPOINT_ROOT / "ensemble_registry_deduplicated.csv")
            registry = registry.sort_values("oof_accuracy", ascending=False).reset_index(drop=True)
            top_model_names = registry["model_name"].head(3).tolist()
            meta_df, meta_test_df = build_meta_frames(registry.head(min(8, len(registry))))

            best_models = ensemble_summary["selected_base_models"]
            best_threshold = float(ensemble_summary["best_threshold"])

            if ensemble_summary["best_meta_family"] == "weighted_average":
                base_weights = np.array(ensemble_summary["best_weights"], dtype=np.float64)
                robust_test_prob = weighted_prob(meta_test_df, best_models, base_weights).astype(np.float32)
            elif ensemble_summary["best_meta_family"] == "rank_average":
                robust_test_prob = rank_average_prob(meta_test_df, best_models).astype(np.float32)
            else:
                _, robust_test_prob = generate_logreg_oof(meta_df, best_models, float(ensemble_summary["best_C"]))

            consensus_cols = list(dict.fromkeys(best_models + top_model_names))[:4]
            consensus_matrix = meta_test_df[consensus_cols].to_numpy(dtype=np.float32)
            consensus_mean = consensus_matrix.mean(axis=1)
            consensus_std = consensus_matrix.std(axis=1)

            pseudo_pos_mask = (consensus_mean >= 0.995) & (consensus_std <= 0.02)
            pseudo_neg_mask = (consensus_mean <= 0.005) & (consensus_std <= 0.02)
            pseudo_mask = pseudo_pos_mask | pseudo_neg_mask
            pseudo_labels = np.where(pseudo_pos_mask, 1, 0).astype(np.int8)

            pseudo_df = pd.DataFrame(
                {
                    "id": test_ids,
                    "consensus_mean": consensus_mean,
                    "consensus_std": consensus_std,
                    "selected": pseudo_mask.astype(np.int8),
                    "pseudo_label": pseudo_labels,
                }
            )
            pseudo_df.to_csv(CHECKPOINT_ROOT / "pseudo_label_candidates.csv", index=False)
            print("Pseudo-labeled rows selected:", int(pseudo_mask.sum()))

            if int(pseudo_mask.sum()) >= 200:
                X_aug = np.vstack([X_all, X_test_all[pseudo_mask]])
                y_aug = np.concatenate([y_all, pseudo_labels[pseudo_mask]])

                pseudo_models = []
                pseudo_test_prob_sum = np.zeros(len(X_test_all), dtype=np.float64)
                chosen_signal_models = [name for name in SIGNAL_CANDIDATES.keys()][: min(3, len(SIGNAL_CANDIDATES))]

                for model_name in chosen_signal_models:
                    params = SIGNAL_CANDIDATES[model_name]
                    X_train_model, _, X_test_model = build_signal_train_eval_test(X_aug, X_aug[:100], X_test_all, params)
                    X_train_model = X_train_model
                    X_test_model = X_test_model
                    model = make_signal_model(params)
                    model.fit(X_train_model, y_aug)
                    test_prob = model.predict_proba(X_test_model)[:, 1]
                    pseudo_test_prob_sum += test_prob
                    pseudo_models.append(model_name + "_pseudo")

                pseudo_test_prob = (pseudo_test_prob_sum / len(chosen_signal_models)).astype(np.float32)

                for alpha, filename in [(0.15, AGGR15_SUBMISSION_NAME), (0.25, AGGR25_SUBMISSION_NAME)]:
                    blend_prob = ((1.0 - alpha) * robust_test_prob + alpha * pseudo_test_prob).astype(np.float32)
                    submission = sample_df.copy()
                    submission["class"] = (blend_prob >= best_threshold).astype(np.int8)
                    submission.to_csv(SUBMISSION_ROOT / filename, index=False)
                    print("Saved aggressive submission:", SUBMISSION_ROOT / filename)

                pseudo_only_submission = sample_df.copy()
                pseudo_only_submission["class"] = (pseudo_test_prob >= best_threshold).astype(np.int8)
                pseudo_only_submission.to_csv(SUBMISSION_ROOT / PURE_PSEUDO_SUBMISSION_NAME, index=False)
                print("Saved pseudo-only submission:", SUBMISSION_ROOT / PURE_PSEUDO_SUBMISSION_NAME)

                pseudo_summary = {
                    "consensus_models": consensus_cols,
                    "pseudo_rows_selected": int(pseudo_mask.sum()),
                    "threshold_used": best_threshold,
                    "generated_submissions": [
                        str(SUBMISSION_ROOT / AGGR15_SUBMISSION_NAME),
                        str(SUBMISSION_ROOT / AGGR25_SUBMISSION_NAME),
                        str(SUBMISSION_ROOT / PURE_PSEUDO_SUBMISSION_NAME),
                    ],
                }
                save_json(OUTPUT_ROOT / "pseudo_label_summary.json", pseudo_summary)
            else:
                print("Pseudo-labeling skipped because too few rows were selected.")
        """
    ),
    code(
        """
        if EXPORT_BUNDLE_AT_END:
            bundle_path = EXPORT_ROOT / "challenge_13_public_lb_push_resume.zip"
            if bundle_path.exists():
                bundle_path.unlink()

            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for folder in [OUTPUT_ROOT, SUBMISSION_ROOT]:
                    if not folder.exists():
                        continue
                    for path in folder.rglob("*"):
                        if path.is_file():
                            zf.write(path, path.relative_to(WORKSPACE_ROOT))
                for name, path in DATA_FILES.items():
                    if path.exists():
                        zf.write(path, path.relative_to(WORKSPACE_ROOT))
            print("Exported bundle:", bundle_path)
        """
    ),
    md(
        """
        ## Recommended submission order

        Orden sugerido para probar en Kaggle:

        1. `challenge_13_public_lb_push_safe_submission.csv`
        2. `challenge_13_public_lb_push_aggressive_alpha15.csv`
        3. `challenge_13_public_lb_push_aggressive_alpha25.csv`

        Si el modo pseudo-labeling no genero filas suficientes, simplemente usa la submission `safe`.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
