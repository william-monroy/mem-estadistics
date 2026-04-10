from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results-after-stack" / "challenge_12_final_stacking_colab_resume" / "output"
TARGET_ROOT = ROOT / "documentacion_proyecto_final" / "graficas_modelos_finales_after_stack"

PUBLIC_SCORES = {
    "signal_features": 0.95000,
    "knn_cleaning": 0.83888,
    "final_stacking": 0.96277,
}

BG = "#000000"
PANEL = "#0B0613"
GRID = "#3B1D63"
TEXT = "#FFFFFF"
MUTED = "#D8B4FE"
PURPLE_PALETTE = ["#6D28D9", "#7C3AED", "#8B5CF6", "#A855F7", "#C084FC", "#DDD6FE"]
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "dark_purple",
    ["#05030A", "#1E1033", "#4C1D95", "#7C3AED", "#C084FC", "#F5EFFF"],
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    sns.set_theme(
        style="darkgrid",
        context="talk",
        rc={
            "axes.facecolor": PANEL,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "legend.facecolor": PANEL,
            "legend.edgecolor": MUTED,
        },
    )
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["savefig.facecolor"] = BG
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = PANEL
    plt.rcParams["font.family"] = "DejaVu Sans"


def style_axis(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(PANEL)
    frame.set_edgecolor(MUTED)
    frame.set_alpha(0.95)
    for text in legend.get_texts():
        text.set_color(TEXT)
    if legend.get_title() is not None:
        legend.get_title().set_color(TEXT)


def save_score_card(output_path: Path, model_label: str, metrics: dict[str, float]) -> None:
    names = list(metrics.keys())
    values = [metrics[name] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = PURPLE_PALETTE[: len(names)]
    bars = ax.bar(names, values, color=palette)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_label} - resumen de accuracy")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT,
        )
    plt.xticks(rotation=12)
    style_axis(ax)
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_confusion_matrix(output_path: Path, model_label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=PURPLE_CMAP, cbar=False, ax=ax, annot_kws={"color": TEXT})
    ax.set_title(f"{model_label} - confusion matrix OOF")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)
    style_axis(ax)
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_probability_distribution(output_path: Path, model_label: str, oof_df: pd.DataFrame) -> None:
    plot_df = oof_df.copy()
    plot_df["Clase real"] = plot_df["y_true"].map({0: "Clase 0", 1: "Clase 1"})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=plot_df,
        x="prob_1",
        hue="Clase real",
        palette={"Clase 0": "#A855F7", "Clase 1": "#DDD6FE"},
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title(f"{model_label} - distribucion de probabilidades OOF")
    ax.set_xlabel("Probabilidad estimada de clase 1")
    ax.set_ylabel("Densidad")
    style_axis(ax)
    style_legend(ax.get_legend())
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_fold_accuracy(output_path: Path, model_label: str, fold_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fold_df, x="fold_idx", y="fold_accuracy", color="#7C3AED", ax=ax)
    ax.set_title(f"{model_label} - accuracy por fold OOF")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    for idx, value in enumerate(fold_df["fold_accuracy"]):
        ax.text(idx, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=10, color=TEXT)
    style_axis(ax)
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_signal_top_candidates(output_path: Path, stage_df: pd.DataFrame) -> None:
    top_df = stage_df.nlargest(10, "cv_mean_accuracy").copy()
    top_df["label"] = top_df.apply(
        lambda row: f"{row['feature__set']} | rawPCA={int(row['feature__raw_pca'])} | {row['scale__method']} | C={row['model__C']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=top_df,
        y="label",
        x="cv_mean_accuracy",
        hue="feature__set",
        dodge=False,
        palette={"all": "#A855F7", "fft": "#7C3AED", "basic": "#DDD6FE"},
        ax=ax,
    )
    ax.set_title("Signal features - top candidatos Stage 2 CV")
    ax.set_xlabel("CV mean accuracy")
    ax.set_ylabel("")
    style_axis(ax)
    style_legend(ax.legend(title="feature__set", loc="lower right"))
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_knn_top_candidates(output_path: Path, stage_df: pd.DataFrame) -> None:
    top_df = stage_df.nlargest(10, "cv_mean_accuracy").copy()
    top_df["label"] = top_df.apply(
        lambda row: f"{row['clean__method']} | {row['scale__method']} | PCA={int(row['pca__n_components'])} | k={int(row['model__n_neighbors'])}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    unique_methods = list(top_df["clean__method"].unique())
    palette = {method: PURPLE_PALETTE[idx % len(PURPLE_PALETTE)] for idx, method in enumerate(unique_methods)}
    sns.barplot(data=top_df, y="label", x="cv_mean_accuracy", hue="clean__method", dodge=False, palette=palette, ax=ax)
    ax.set_title("KNN cleaning - top candidatos Stage 3 CV")
    ax.set_xlabel("CV mean accuracy")
    ax.set_ylabel("")
    style_axis(ax)
    style_legend(ax.legend(title="clean__method", loc="lower right"))
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_knn_group_visualization(output_path: Path, summary: dict, oof_df: pd.DataFrame) -> None:
    data_path = ROOT / "data" / "training.csv"
    train_df = pd.read_csv(data_path)
    feature_cols = [c for c in train_df.columns if c.startswith("V")]
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["class"].to_numpy(dtype=np.int8)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_2d = PCA(n_components=2, random_state=301655)
    X_2d = pca_2d.fit_transform(X_scaled)

    params = summary["best_params"]
    knn_2d = KNeighborsClassifier(
        n_neighbors=int(params["model__n_neighbors"]),
        metric=params["model__metric"],
        weights=params["model__weights"],
    )
    knn_2d.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = knn_2d.predict(grid).reshape(xx.shape)

    plot_df = pd.DataFrame(
        {
            "pc1": X_2d[:, 0],
            "pc2": X_2d[:, 1],
            "y_true": y,
        }
    ).merge(
        oof_df[["id", "pred"]].rename(columns={"pred": "oof_pred"}),
        left_index=True,
        right_index=True,
        how="left",
    )
    plot_df["correct_oof"] = (plot_df["y_true"] == plot_df["oof_pred"]).fillna(True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap_bg = LinearSegmentedColormap.from_list("knn_bg", ["#12071F", "#2E1065", "#5B21B6"])
    axes[0].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.55)
    sns.scatterplot(
        data=plot_df.sample(min(len(plot_df), 3000), random_state=301655),
        x="pc1",
        y="pc2",
        hue="y_true",
        palette={0: "#C084FC", 1: "#F5EFFF"},
        alpha=0.65,
        s=22,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("KNN en PCA 2D: frontera aproximada")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    style_axis(axes[0])
    style_legend(axes[0].legend(title="Clase real", loc="lower right"))

    sampled = plot_df.sample(min(len(plot_df), 3000), random_state=301656).copy()
    sns.scatterplot(
        data=sampled,
        x="pc1",
        y="pc2",
        hue="y_true",
        style="correct_oof",
        palette={0: "#A855F7", 1: "#DDD6FE"},
        markers={True: "o", False: "X"},
        alpha=0.75,
        s=28,
        ax=axes[1],
    )
    axes[1].set_title("KNN en PCA 2D: clases y acierto OOF")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    style_axis(axes[1])
    style_legend(axes[1].legend(title="Clase / acierto", loc="lower right"))

    note = (
        "Visualizacion 2D aproximada: usa PCA a 2 componentes y un KNN reentrenado en ese espacio.\n"
        "Sirve para interpretar vecindad y separacion, no para representar exactamente el modelo final en 64 dimensiones."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", color=MUTED, fontsize=10)
    fig.savefig(output_path, dpi=220, facecolor=BG)
    plt.close(fig)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=np.abs(den) > 1e-12)


def build_signal_feature_bank(X_input: np.ndarray) -> dict[str, np.ndarray]:
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


def choose_signal_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown signal feature scaler: {name}")


def save_signal_group_visualization(output_path: Path, summary: dict, oof_df: pd.DataFrame) -> None:
    data_path = ROOT / "data" / "training.csv"
    train_df = pd.read_csv(data_path)
    feature_cols = [c for c in train_df.columns if c.startswith("V")]
    X_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["class"].to_numpy(dtype=np.int8)

    params = summary["best_params"]
    feature_bank = build_signal_feature_bank(X_raw)
    X_base = feature_bank[params["feature__set"]].astype(np.float32)

    raw_pca_dim = int(params["feature__raw_pca"])
    if raw_pca_dim > 0:
        raw_scaler = StandardScaler()
        X_raw_scaled = raw_scaler.fit_transform(X_raw)
        raw_pca = PCA(n_components=min(raw_pca_dim, X_raw_scaled.shape[0], X_raw_scaled.shape[1]), random_state=301655)
        X_raw_pca = raw_pca.fit_transform(X_raw_scaled).astype(np.float32)
        X_model = np.hstack([X_base, X_raw_pca]).astype(np.float32)
    else:
        X_model = X_base

    model_scaler = choose_signal_scaler(params["scale__method"])
    X_model_scaled = model_scaler.fit_transform(X_model)

    pca_2d = PCA(n_components=2, random_state=301655)
    X_2d = pca_2d.fit_transform(X_model_scaled)

    svm_2d = SVC(
        C=float(params["model__C"]),
        gamma=float(params["model__gamma"]),
        kernel="rbf",
        probability=False,
    )
    svm_2d.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 0.8, X_2d[:, 0].max() + 0.8
    y_min, y_max = X_2d[:, 1].min() - 0.8, X_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = svm_2d.predict(grid).reshape(xx.shape)

    plot_df = pd.DataFrame(
        {
            "pc1": X_2d[:, 0],
            "pc2": X_2d[:, 1],
            "y_true": y,
        }
    ).merge(
        oof_df[["id", "pred"]].rename(columns={"pred": "oof_pred"}),
        left_index=True,
        right_index=True,
        how="left",
    )
    plot_df["correct_oof"] = (plot_df["y_true"] == plot_df["oof_pred"]).fillna(True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap_bg = LinearSegmentedColormap.from_list("signal_bg", ["#12071F", "#3B0764", "#6D28D9"])
    axes[0].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.58)
    sns.scatterplot(
        data=plot_df.sample(min(len(plot_df), 3000), random_state=301655),
        x="pc1",
        y="pc2",
        hue="y_true",
        palette={0: "#C084FC", 1: "#F5EFFF"},
        alpha=0.65,
        s=22,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Signal features en PCA 2D: frontera aproximada")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    style_axis(axes[0])
    style_legend(axes[0].legend(title="Clase real", loc="lower right"))

    sampled = plot_df.sample(min(len(plot_df), 3000), random_state=301657).copy()
    sns.scatterplot(
        data=sampled,
        x="pc1",
        y="pc2",
        hue="y_true",
        style="correct_oof",
        palette={0: "#A855F7", 1: "#DDD6FE"},
        markers={True: "o", False: "X"},
        alpha=0.75,
        s=28,
        ax=axes[1],
    )
    axes[1].set_title("Signal features en PCA 2D: clases y acierto OOF")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    style_axis(axes[1])
    style_legend(axes[1].legend(title="Clase / acierto", loc="lower right"))

    note = (
        "Visualizacion 2D aproximada: usa el espacio exacto de signal features del mejor modelo,\n"
        "lo proyecta a PCA 2D y reentrena un SVM-RBF en ese plano para interpretar separacion y errores."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", color=MUTED, fontsize=10)
    fig.savefig(output_path, dpi=220, facecolor=BG)
    plt.close(fig)


def save_stacking_top_candidates(output_path: Path, stage_df: pd.DataFrame) -> None:
    top_df = stage_df.nlargest(12, "meta_oof_accuracy").copy()
    top_df["short_label"] = top_df.apply(
        lambda row: f"{row['meta_family']} | thr={row['best_threshold']:.2f}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top_df,
        y="short_label",
        x="meta_oof_accuracy",
        hue="meta_family",
        dodge=False,
        palette={"weighted_average": "#A855F7", "logreg": "#DDD6FE", "rank_average": "#7C3AED"},
        ax=ax,
    )
    ax.set_title("Final stacking - mejores combinaciones")
    ax.set_xlabel("Meta OOF accuracy")
    ax.set_ylabel("")
    style_axis(ax)
    style_legend(ax.legend(title="meta_family", loc="lower right"))
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_best_blend_weights(output_path: Path, summary_path: Path, base_summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    base_df = pd.read_csv(base_summary_path)
    weights = summary["best_candidate"]["weights"]
    model_names = summary["best_base_models"]

    plot_df = pd.DataFrame(
        {
            "model_name": model_names,
            "weight": weights,
        }
    ).merge(base_df[["model_name", "oof_accuracy"]], on="model_name", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=plot_df, x="model_name", y="weight", hue="model_name", palette=PURPLE_PALETTE[: len(plot_df)], legend=False, ax=axes[0])
    axes[0].set_title("Pesos del mejor blend")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Peso")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=plot_df, x="model_name", y="oof_accuracy", hue="model_name", palette=list(reversed(PURPLE_PALETTE[: len(plot_df)])), legend=False, ax=axes[1])
    axes[1].set_title("OOF accuracy de modelos base")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("OOF accuracy")
    axes[1].tick_params(axis="x", rotation=15)

    for ax in axes:
        style_axis(ax)
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def save_stacking_meta_space_visualization(
    output_path: Path,
    summary_path: Path,
    knn_oof_path: Path,
    signal_oof_path: Path,
    meta_oof_path: Path,
) -> None:
    summary = json.loads(summary_path.read_text())
    knn_df = pd.read_csv(knn_oof_path)[["id", "y_true", "prob_1"]].rename(columns={"prob_1": "knn_prob"})
    signal_df = pd.read_csv(signal_oof_path)[["id", "prob_1"]].rename(columns={"prob_1": "signal_prob"})
    meta_df = pd.read_csv(meta_oof_path)[["id", "prob_1", "pred"]].rename(columns={"prob_1": "meta_prob", "pred": "meta_pred"})

    plot_df = knn_df.merge(signal_df, on="id", how="inner").merge(meta_df, on="id", how="inner")
    plot_df["correct_meta"] = plot_df["y_true"] == plot_df["meta_pred"]

    weights = summary["best_candidate"]["weights"]
    threshold = float(summary["best_threshold"])

    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, 240),
        np.linspace(0.0, 1.0, 240),
    )
    zz = (weights[0] * xx + weights[1] * yy >= threshold).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap_bg = LinearSegmentedColormap.from_list("stack_bg", ["#130818", "#34115A", "#7C3AED"])
    axes[0].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.58)
    sns.scatterplot(
        data=plot_df,
        x="knn_prob",
        y="signal_prob",
        hue="y_true",
        palette={0: "#C084FC", 1: "#F5EFFF"},
        alpha=0.62,
        s=22,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Stacking: espacio meta exacto de probabilidades")
    axes[0].set_xlabel("Probabilidad KNN cleaning")
    axes[0].set_ylabel("Probabilidad signal features")
    style_axis(axes[0])
    style_legend(axes[0].legend(title="Clase real", loc="lower right"))

    sns.scatterplot(
        data=plot_df,
        x="knn_prob",
        y="signal_prob",
        hue="y_true",
        style="correct_meta",
        palette={0: "#A855F7", 1: "#DDD6FE"},
        markers={True: "o", False: "X"},
        alpha=0.75,
        s=28,
        ax=axes[1],
    )
    axes[1].axline((0, threshold / weights[1]), slope=-(weights[0] / weights[1]), color="#F5EFFF", linestyle="--", linewidth=1.5)
    axes[1].set_title("Stacking: aciertos y errores sobre la frontera final")
    axes[1].set_xlabel("Probabilidad KNN cleaning")
    axes[1].set_ylabel("Probabilidad signal features")
    style_axis(axes[1])
    style_legend(axes[1].legend(title="Clase / acierto", loc="lower right"))

    note = (
        f"Visualizacion exacta del meta-modelo final: weighted average con pesos {weights[0]:.3f} y {weights[1]:.3f}, "
        f"threshold = {threshold:.2f}."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", color=MUTED, fontsize=10)
    fig.savefig(output_path, dpi=220, facecolor=BG)
    plt.close(fig)


def save_comparison_chart(output_path: Path) -> None:
    signal_summary = json.loads((RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "summary.json").read_text())
    knn_summary = json.loads((RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "summary.json").read_text())
    stack_summary = json.loads((RESULTS_ROOT / "challenge_12_final_stacking_colab" / "summary.json").read_text())

    rows = [
        {"model": "Signal features", "metric": "Validation", "value": signal_summary["validation_accuracy"]},
        {"model": "Signal features", "metric": "OOF", "value": signal_summary["oof_accuracy"]},
        {"model": "Signal features", "metric": "Public", "value": PUBLIC_SCORES["signal_features"]},
        {"model": "KNN cleaning", "metric": "Validation", "value": knn_summary["validation_accuracy"]},
        {"model": "KNN cleaning", "metric": "OOF", "value": knn_summary["oof_accuracy"]},
        {"model": "KNN cleaning", "metric": "Public", "value": PUBLIC_SCORES["knn_cleaning"]},
        {"model": "Final stacking", "metric": "Meta OOF", "value": stack_summary["best_meta_oof_accuracy"]},
        {"model": "Final stacking", "metric": "Public", "value": PUBLIC_SCORES["final_stacking"]},
    ]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    metric_palette = {
        "Validation": "#7C3AED",
        "OOF": "#A855F7",
        "Public": "#DDD6FE",
        "Meta OOF": "#C084FC",
    }
    sns.barplot(data=df, x="model", y="value", hue="metric", palette=metric_palette, ax=ax)
    ax.set_title("Comparacion global de los tres modelos finales")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.75, 1.0)
    style_axis(ax)
    style_legend(ax.legend(title="Metrica", loc="lower right"))
    fig.savefig(output_path, dpi=200, facecolor=BG)
    plt.close(fig)


def generate_signal_graphs() -> dict[str, float]:
    output_dir = TARGET_ROOT / "signal_features"
    ensure_dir(output_dir)

    summary_path = RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "summary.json"
    oof_path = RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "oof_probabilities.csv"
    fold_path = RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "oof_fold_summary.csv"
    stage_path = RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "checkpoints" / "stage2_cv_summary.csv"

    summary = json.loads(summary_path.read_text())
    oof_df = pd.read_csv(oof_path)
    fold_df = pd.read_csv(fold_path)
    stage_df = pd.read_csv(stage_path)

    save_score_card(
        output_dir / "01_score_card.png",
        "Signal features",
        {
            "Validation": summary["validation_accuracy"],
            "OOF": summary["oof_accuracy"],
            "Public": PUBLIC_SCORES["signal_features"],
        },
    )
    save_confusion_matrix(output_dir / "02_oof_confusion_matrix.png", "Signal features", oof_df["y_true"].to_numpy(), oof_df["pred"].to_numpy())
    save_probability_distribution(output_dir / "03_oof_probability_distribution.png", "Signal features", oof_df)
    save_fold_accuracy(output_dir / "04_oof_fold_accuracy.png", "Signal features", fold_df)
    save_signal_top_candidates(output_dir / "05_top_stage2_candidates.png", stage_df)
    save_signal_group_visualization(output_dir / "06_signal_group_visualization_pca2d.png", summary, oof_df)

    return {
        "model": "signal_features",
        "validation_accuracy": summary["validation_accuracy"],
        "oof_accuracy": summary["oof_accuracy"],
        "public_score": PUBLIC_SCORES["signal_features"],
    }


def generate_knn_graphs() -> dict[str, float]:
    output_dir = TARGET_ROOT / "knn_cleaning"
    ensure_dir(output_dir)

    summary_path = RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "summary.json"
    oof_path = RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "oof_probabilities.csv"
    fold_path = RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "oof_fold_summary.csv"
    stage_path = RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "checkpoints" / "stage3_local_cv_summary.csv"

    summary = json.loads(summary_path.read_text())
    oof_df = pd.read_csv(oof_path)
    fold_df = pd.read_csv(fold_path)
    stage_df = pd.read_csv(stage_path)

    save_score_card(
        output_dir / "01_score_card.png",
        "KNN cleaning",
        {
            "Validation": summary["validation_accuracy"],
            "OOF": summary["oof_accuracy"],
            "Public": PUBLIC_SCORES["knn_cleaning"],
        },
    )
    save_confusion_matrix(output_dir / "02_oof_confusion_matrix.png", "KNN cleaning", oof_df["y_true"].to_numpy(), oof_df["pred"].to_numpy())
    save_probability_distribution(output_dir / "03_oof_probability_distribution.png", "KNN cleaning", oof_df)
    save_fold_accuracy(output_dir / "04_oof_fold_accuracy.png", "KNN cleaning", fold_df)
    save_knn_top_candidates(output_dir / "05_top_stage3_candidates.png", stage_df)
    save_knn_group_visualization(output_dir / "06_knn_group_visualization_pca2d.png", summary, oof_df)

    return {
        "model": "knn_cleaning",
        "validation_accuracy": summary["validation_accuracy"],
        "oof_accuracy": summary["oof_accuracy"],
        "public_score": PUBLIC_SCORES["knn_cleaning"],
    }


def generate_stacking_graphs() -> dict[str, float]:
    output_dir = TARGET_ROOT / "final_stacking"
    ensure_dir(output_dir)

    summary_path = RESULTS_ROOT / "challenge_12_final_stacking_colab" / "summary.json"
    oof_path = RESULTS_ROOT / "challenge_12_final_stacking_colab" / "meta_oof_probabilities.csv"
    stage_path = RESULTS_ROOT / "challenge_12_final_stacking_colab" / "checkpoints" / "final_stacking_search_results.csv"
    base_summary_path = RESULTS_ROOT / "challenge_12_final_stacking_colab" / "checkpoints" / "base_model_summary.csv"
    knn_oof_path = RESULTS_ROOT / "challenge_08_knn_cleaning_colab_ultra" / "oof_probabilities.csv"
    signal_oof_path = RESULTS_ROOT / "challenge_10_signal_features_colab_ultra" / "oof_probabilities.csv"

    summary = json.loads(summary_path.read_text())
    oof_df = pd.read_csv(oof_path)
    stage_df = pd.read_csv(stage_path)

    save_score_card(
        output_dir / "01_score_card.png",
        "Final stacking",
        {
            "Meta OOF": summary["best_meta_oof_accuracy"],
            "Public": PUBLIC_SCORES["final_stacking"],
        },
    )
    save_confusion_matrix(output_dir / "02_oof_confusion_matrix.png", "Final stacking", oof_df["y_true"].to_numpy(), oof_df["pred"].to_numpy())
    save_probability_distribution(output_dir / "03_oof_probability_distribution.png", "Final stacking", oof_df)
    save_stacking_top_candidates(output_dir / "04_top_stacking_candidates.png", stage_df)
    save_best_blend_weights(output_dir / "05_best_blend_weights.png", summary_path, base_summary_path)
    save_stacking_meta_space_visualization(
        output_dir / "06_stacking_meta_space_visualization.png",
        summary_path,
        knn_oof_path,
        signal_oof_path,
        oof_path,
    )

    return {
        "model": "final_stacking",
        "meta_oof_accuracy": summary["best_meta_oof_accuracy"],
        "public_score": PUBLIC_SCORES["final_stacking"],
    }


def write_readme(metrics_df: pd.DataFrame) -> None:
    readme_path = TARGET_ROOT / "README.md"
    summary_lines = metrics_df.fillna("").to_csv(index=False).strip()
    text = f"""# Graficas Modelos Finales After Stack

Version visual: `dark mode`

- fondo negro
- texto blanco
- paletas en tonos morados cuando aplica

Esta carpeta contiene graficas nuevas generadas a partir de:

- `challenge/results-after-stack/challenge_12_final_stacking_colab_resume/output/...`
- los public scores observados manualmente en Kaggle

Modelos cubiertos:

- `signal_features`
- `knn_cleaning`
- `final_stacking`

## Estructura

- `comparison/00_model_score_comparison.png`
- `signal_features/`
- `knn_cleaning/`
- `final_stacking/`

## Notas metodologicas

- Para `signal_features` y `knn_cleaning`, la confusion matrix y la distribucion de probabilidades se construyeron con `oof_probabilities.csv`.
- Para `final_stacking`, se uso `meta_oof_probabilities.csv`.
- Para `signal_features` y `knn_cleaning`, `pred` usa el threshold persistido en el artefacto OOF.
- Para `final_stacking`, `pred` ya esta persistido con el mejor threshold encontrado (`0.49`).
- La figura `06_signal_group_visualization_pca2d.png` es una visualizacion aproximada del mejor modelo de `signal_features` en 2D usando el espacio de features ingenierizadas y una proyeccion PCA.
- La figura `06_knn_group_visualization_pca2d.png` es una visualizacion aproximada del comportamiento de KNN en 2D usando PCA. No representa exactamente la frontera del modelo final en 64 dimensiones.
- La figura `06_stacking_meta_space_visualization.png` es una visualizacion exacta del modelo final de stacking en el plano de probabilidades de sus dos modelos base.
- Los public scores usados en estas graficas son:
  - `signal_features = 0.95000`
  - `knn_cleaning = 0.83888`
  - `final_stacking = 0.96277`

## Resumen numerico

```csv
{summary_lines}
```
"""
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dir(TARGET_ROOT)
    ensure_dir(TARGET_ROOT / "comparison")

    rows = [
        generate_signal_graphs(),
        generate_knn_graphs(),
        generate_stacking_graphs(),
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TARGET_ROOT / "metrics_summary.csv", index=False)
    save_comparison_chart(TARGET_ROOT / "comparison" / "00_model_score_comparison.png")
    write_readme(metrics_df)
    print(f"Graphs written to {TARGET_ROOT}")


if __name__ == "__main__":
    main()
