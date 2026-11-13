"""Türkçe, 200 dpi PNG grafikler ve gerçek modelden hesaplanan karar sınırları."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from models import LinearRegressionScratch
from .metrics import confusion_matrix
from .preprocessing import feature_transform

COLORS = ["#2878b5", "#dd7830", "#39936c", "#8b63aa", "#737b85"]
CLASS_COLORS = COLORS[:2]
GRAPH_NAMES = ["two_moons_scatter", "class_distribution", "linear_training_loss", "logistic_training_loss",
               "linear_confusion_matrix", "logistic_confusion_matrix", "metrics_comparison", "decision_boundaries",
               "original_vs_polynomial_boundaries", "original_vs_polynomial_metrics", "polynomial_training_losses",
               "polynomial_confusion_matrices"]


def save_figure(fig, directory, name):
    Path(directory).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(directory) / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset(df, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for label, color in enumerate(CLASS_COLORS):
        subset = df[df["label"] == label]
        ax.scatter(subset["x1"], subset["x2"], color=color, alpha=0.75, s=25, label=f"Sınıf {label}")
    ax.set(title="Two Moons — Gerçek Sınıfların Dağılımı", xlabel="x₁", ylabel="x₂", aspect="equal")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, directory, "two_moons_scatter")
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"Sınıf {i}" for i in counts.index], counts, color=CLASS_COLORS)
    ax.bar_label(bars, padding=4)
    ax.set(title="Sınıf Dağılımı", ylabel="Örnek sayısı", ylim=(0, max(counts) * 1.15))
    save_figure(fig, directory, "class_distribution")


def decision_scores(pipeline, raw_X):
    """Izgara da testle aynı dönüşüm ve eğitim ölçeklerini kullanır."""
    transformed = feature_transform(raw_X, pipeline["polynomial"])
    scaled = pipeline["scaler"].transform(transformed)
    model = pipeline["model"]
    return model.predict(scaled) if isinstance(model, LinearRegressionScratch) else model.predict_proba(scaled)


def plot_boundaries(pipelines, X, y, train_indices, test_indices, directory):
    x_values = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 280)
    y_values = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 240)
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    names = list(pipelines)
    grid_scores = {name: decision_scores(pipelines[name], grid).reshape(xx.shape) for name in names}
    np.savez_compressed(Path(directory).parent / "decision_grid.npz", x_values=x_values, y_values=y_values,
                        **{f"scores_{i + 1}": grid_scores[name] for i, name in enumerate(names)})
    legend = [Line2D([], [], marker="o", linestyle="", color=color, label=f"Gerçek sınıf {label}")
              for label, color in enumerate(CLASS_COLORS)]
    legend += [Line2D([], [], marker="o", color="gray", linestyle="", label="Eğitim örneği"),
               Line2D([], [], marker="^", markeredgecolor="black", color="gray", linestyle="", label="Test örneği"),
               Line2D([], [], color="black", label="Modelin 0,5 sınırı")]
    for order, shape, filename, title in [
        (names[:2], (1, 2), "decision_boundaries", "Ana Karşılaştırma — Özgün Özelliklerde Karar Sınırları"),
        ([names[0], names[2], names[1], names[3]], (2, 2), "original_vs_polynomial_boundaries",
         "Ek Deney — Özgün ve Polinom Özelliklerde Gerçek Model Sınırları"),
    ]:
        fig, axes = plt.subplots(*shape, figsize=(14, 5.5 * shape[0]), squeeze=False, sharex=True, sharey=True,
                                 layout="constrained")
        fig.get_layout_engine().set(h_pad=0.10, w_pad=0.08, hspace=0.10)
        for ax, name in zip(axes.ravel(), order):
            scores = grid_scores[name]
            ax.contourf(xx, yy, (scores >= 0.5).astype(int), levels=[-0.5, 0.5, 1.5],
                        cmap=ListedColormap(CLASS_COLORS), alpha=0.14)
            if scores.min() < 0.5 < scores.max():
                ax.contour(xx, yy, scores, levels=[0.5], colors="black", linewidths=1.6)
            for indices, marker in [(train_indices, "o"), (test_indices, "^")]:
                for label, color in enumerate(CLASS_COLORS):
                    subset = indices[y[indices] == label]
                    ax.scatter(X[subset, 0], X[subset, 1], color=color, marker=marker,
                               s=25 if marker == "o" else 48, alpha=0.8,
                               edgecolors="black" if marker == "^" else "white", linewidths=0.45)
            ax.set(title=name, xlabel="x₁ (özgün ölçek)", ylabel="x₂ (özgün ölçek)", aspect="equal")
            ax.grid(alpha=0.15)
        fig.suptitle(title)
        fig.legend(handles=legend, loc="outside lower center", ncol=5, fontsize=9)
        # Constrained layout, iki satırın başlıkları ve ortak lejant için yer ayırır.
        fig.savefig(Path(directory) / f"{filename}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_results(pipelines, metrics, y_test, predictions, directory):
    names = list(pipelines)
    for name, filename, title in [(names[0], "linear_training_loss", "Doğrusal Regresyon — Eğitim Kaybı"),
                                  (names[1], "logistic_training_loss", "Lojistik Regresyon — Eğitim Kaybı")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(pipelines[name]["model"].loss_history_, color=COLORS[names.index(name)])
        ax.set(title=title, xlabel="Gradient Descent adımı", ylabel="J(θ): MSE / 2" if name == names[0] else "J(θ): BCE")
        ax.grid(alpha=0.2)
        save_figure(fig, directory, filename)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, names[2:]):
        ax.plot(pipelines[name]["model"].loss_history_, color=COLORS[names.index(name)])
        ax.set(title=name, xlabel="Gradient Descent adımı", ylabel="MSE / 2" if name == names[2] else "BCE")
        ax.grid(alpha=0.2)
    fig.suptitle("Ek Deney — Polinom Modellerin Eğitim Kayıpları")
    save_figure(fig, directory, "polynomial_training_losses")

    for keys, filename in [([names[0]], "linear_confusion_matrix"), ([names[1]], "logistic_confusion_matrix"),
                           (names[2:], "polynomial_confusion_matrices")]:
        fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.8), squeeze=False)
        for ax, name in zip(axes.ravel(), keys):
            matrix = confusion_matrix(y_test, predictions[name])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=len(y_test))
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, f"{['TN', 'FP', 'FN', 'TP'][row * 2 + col]}\n{matrix[row, col]}",
                            ha="center", va="center", fontsize=15)
            ax.set_xticks([0, 1], ["Sınıf 0", "Sınıf 1"])
            ax.set_yticks([0, 1], ["Sınıf 0", "Sınıf 1"])
            ax.set(title=f"{name}\nTest Karmaşıklık Matrisi", xlabel="Tahmin", ylabel="Gerçek")
        save_figure(fig, directory, filename)

    fields = ["accuracy", "precision", "recall", "f1"]
    for keys, filename, title in [
        (names[:2] + ["Çoğunluk referansı"], "metrics_comparison", "Ana Karşılaştırma ve Çoğunluk Referansı"),
        (names, "original_vs_polynomial_metrics", "Ek Deney — Özgün ve Polinom Özellikler"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        positions, width = np.arange(4), 0.8 / len(keys)
        for i, key in enumerate(keys):
            color = COLORS[names.index(key)] if key in names else COLORS[4]
            bars = ax.bar(positions - 0.4 + width / 2 + i * width,
                          metrics.loc[key, fields].to_numpy(dtype=float), width=width, label=key, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_xticks(positions, ["Doğruluk", "Kesinlik", "Duyarlılık", "F1"])
        ax.set(title=title, ylabel="Skor", ylim=(0, 1.12))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, directory, filename)
