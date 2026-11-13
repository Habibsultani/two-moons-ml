"""Kaydedilen parametre, test tahminleri ve karar ızgarasının bağımsız denetimi."""

from collections import Counter
import hashlib
import json
import sys

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from main import EXPERIMENTS, ROOT, SEED
from models import LinearRegressionScratch, LogisticRegressionScratch
from utils.preprocessing import FEATURE_NAMES, POLYNOMIAL_NAMES, feature_transform, load_dataset, stratified_split
from utils.visualization import GRAPH_NAMES


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_metrics(y, predicted):
    """Projedeki metrik işlevini çağırmadan doğrudan sayar."""
    counts = Counter(zip(y, predicted))
    tn, fp, fn, tp = [counts[pair] for pair in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    def ratio(a, b):
        return a / b if b else 0.
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": (tn + tp) / len(y),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": ratio(tn, tn + fp)}


def verify():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = ROOT / "results"
    df, actual_info = load_dataset(ROOT / "data/two_moons.csv")
    info = json.loads((results / "dataset_info.json").read_text(encoding="utf-8"))
    require(info["sha256"] == actual_info["sha256"], "Veri dosyasının hash değeri değişmiş.")
    X, y = df[FEATURE_NAMES].to_numpy(), df["label"].to_numpy(dtype=int)
    train, test = stratified_split(y, seed=SEED)
    inner_fit, inner_val = stratified_split(y[train], seed=SEED)
    fitting, val = train[inner_fit], train[inner_val]
    require(not set(train) & set(test), "Eğitim/test kesişimi bulundu.")
    require(not set(fitting) & set(val), "İç eğitim/doğrulama kesişimi bulundu.")
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(y)))
    np.testing.assert_array_equal(np.bincount(y[test]), [30, 30])
    split = pd.read_csv(results / "split_indices.csv").set_index("source_row")
    for indices, name in [(fitting, "inner_train"), (val, "validation"), (test, "test")]:
        require((split.loc[indices + 1, "split"] == name).all(), f"Bölme kayıtları uyuşmuyor: {name}")
    predictions = pd.read_csv(results / "test_predictions.csv")
    np.testing.assert_array_equal(predictions["source_row"], test + 1)
    np.testing.assert_array_equal(predictions["y_true"], y[test])
    metrics = pd.read_csv(results / "metrics.csv", index_col="model")
    validation = pd.read_csv(results / "validation_results.csv")
    selected = json.loads((results / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    require(len(validation) == 36, "36 doğrulama adayı bekleniyor.")
    with np.load(results / "model_parameters.npz") as saved, np.load(results / "decision_grid.npz") as grid:
        for key, indices in [("train_indices", train), ("test_indices", test),
                             ("inner_train_indices", fitting), ("validation_indices", val)]:
            np.testing.assert_array_equal(saved[key], indices)
        xx, yy = np.meshgrid(grid["x_values"], grid["y_values"])
        raw_grid = np.column_stack([xx.ravel(), yy.ravel()])
        for i, (name, cls, polynomial) in enumerate(EXPERIMENTS, start=1):
            raw_train = feature_transform(X[train], polynomial)
            expected_mean, expected_std = raw_train.mean(axis=0), raw_train.std(axis=0, ddof=0)
            expected_std = np.where(expected_std == 0, 1., expected_std)
            np.testing.assert_allclose(saved[f"mean_{i}"], expected_mean)
            np.testing.assert_allclose(saved[f"std_{i}"], expected_std)
            parameters = pd.read_csv(results / f"parameters_{i}.csv")
            require(parameters["feature"].tolist() == (POLYNOMIAL_NAMES if polynomial else FEATURE_NAMES), "Özellik sırası yanlış.")
            np.testing.assert_allclose(parameters["theta"], saved[f"theta_{i}"])
            for raw_X, expected in [(X[test], predictions[f"{name} / raw"].to_numpy()),
                                    (raw_grid, grid[f"scores_{i}"].ravel())]:
                scaled = (feature_transform(raw_X, polynomial) - saved[f"mean_{i}"]) / saved[f"std_{i}"]
                z = scaled @ saved[f"theta_{i}"] + saved[f"bias_{i}"]
                raw = z if cls is LinearRegressionScratch else LogisticRegressionScratch.sigmoid(z)
                np.testing.assert_allclose(raw, expected, atol=1e-12)
            predicted = (predictions[f"{name} / raw"].to_numpy() >= 0.5).astype(int)
            np.testing.assert_array_equal(predicted, predictions[f"{name} / class"])
            for key, value in independent_metrics(y[test], predicted).items():
                np.testing.assert_allclose(value, metrics.loc[name, key], atol=1e-14)
            table = validation[validation["model"] == name]
            require(len(table) == 9, "Her yapılandırmada 9 aday bekleniyor.")
            best = table.sort_values(["f1", "accuracy", "epochs", "learning_rate"], ascending=[False, False, True, True]).iloc[0]
            require(selected[name]["epochs"] == best["epochs"] and selected[name]["learning_rate"] == best["learning_rate"],
                    "Doğrulama seçim kuralı uyuşmuyor.")
            history = pd.read_csv(results / f"loss_history_{i}.csv")["training_loss"].to_numpy()
            require(len(history) == selected[name]["epochs"] + 1 and np.isfinite(history).all(), "Kayıp geçmişi geçersiz.")
            require(np.all(np.diff(history) <= 1e-12), "Eğitim kaybında beklenmeyen artış bulundu.")
            Xt = (raw_train - expected_mean) / expected_std
            loss, dw, db = cls.loss_and_gradients(Xt, y[train], saved[f"theta_{i}"], saved[f"bias_{i}"])
            np.testing.assert_allclose(loss, history[-1])
            np.testing.assert_allclose(loss, metrics.loc[name, "train_loss"])
            np.testing.assert_allclose(np.linalg.norm(np.r_[dw, db]), metrics.loc[name, "final_gradient_l2"])
    majority = int(np.argmax(np.bincount(y[train], minlength=2)))
    baseline = np.full(len(test), majority)
    np.testing.assert_array_equal(baseline, predictions["Çoğunluk referansı / class"])
    for key, value in independent_metrics(y[test], baseline).items():
        np.testing.assert_allclose(value, metrics.loc["Çoğunluk referansı", key])
    certificate = info["geometry_check"]
    if certificate["found"]:
        points = []
        for label in [0, 1]:
            indices = np.array(certificate[f"class_{label}_rows"]) - 1
            require(set(indices).issubset(set(train)), "Geometri tanığında test satırı kullanılmış.")
            np.testing.assert_array_equal(y[indices], [label, label])
            weights = np.array(certificate[f"class_{label}_weights"])
            require(np.all(weights > 0), "Geometri ağırlıkları pozitif olmalı.")
            np.testing.assert_allclose(weights.sum(), 1.)
            points.append(weights @ X[indices])
        np.testing.assert_allclose(points[0], points[1], atol=1e-12)
        np.testing.assert_allclose(points[0], certificate["intersection"], atol=1e-12)
    for name in GRAPH_NAMES:
        pixels = mpimg.imread(results / "graphs" / f"{name}.png")
        require(min(pixels.shape[:2]) > 500 and np.std(pixels[:, :, :3]) > 0.01, f"Grafik boş/küçük: {name}")
    summary = (
        f"BAŞARILI: Veri SHA256={hashlib.sha256((ROOT / 'data/two_moons.csv').read_bytes()).hexdigest()}.\n"
        "Dört modelin aynı 60 test satırı, eğitimden öğrenilen ölçekler, doğrulama seçimi, "
        "bağımsız metrikler, kayıp kayıtları, parametrelerden test tahminleri ve karar ızgarası doğrulandı.\n"
        f"{len(GRAPH_NAMES)} PNG dosyası okunabilir ve boş değil; geometri tanığı da kontrol edildi.\n"
    )
    (results / "verification_summary.txt").write_text(summary, encoding="utf-8")
    print("Doğrulama başarılı: 4 model, 60 ortak test örneği, 36 aday, 12 grafik ve gerçek karar ızgarası.")


if __name__ == "__main__":
    verify()
