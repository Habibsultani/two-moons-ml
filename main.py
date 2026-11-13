"""Two Moons: sıfırdan iki algoritma, ortak test ve ayrı polinom ek deneyi."""

from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from models import LinearRegressionScratch, LogisticRegressionScratch
from utils.geometry import linear_nonseparability_certificate
from utils.metrics import classification_metrics
from utils.preprocessing import (FEATURE_NAMES, POLYNOMIAL_NAMES, StandardScalerScratch,
                                 encode_labels, feature_transform, load_dataset, stratified_split)
from utils.reporting import write_summary
from utils.visualization import plot_boundaries, plot_dataset, plot_results

ROOT = Path(__file__).resolve().parent
SEED = 42
# Bu ızgara test sonuçları görülmeden belirlenir ve dört deneyde aynıdır.
LEARNING_RATES = (0.01, 0.05, 0.1)
EPOCHS = (1000, 3000, 6000)
EXPERIMENTS = [
    ("Doğrusal / Özgün", LinearRegressionScratch, False),
    ("Lojistik / Özgün", LogisticRegressionScratch, False),
    ("Doğrusal / Polinom", LinearRegressionScratch, True),
    ("Lojistik / Polinom", LogisticRegressionScratch, True),
]


def predict_classes(model, X):
    return model.predict_classes(X) if isinstance(model, LinearRegressionScratch) else model.predict(X)


def select_configuration(cls, X_fit, y_fit, X_val, y_val, name):
    """Yalnızca iç eğitim/doğrulama görür; test bu işlevin girdisi değildir."""
    best_key, best, rows = None, None, []
    for rate in LEARNING_RATES:
        for epochs in EPOCHS:
            model = cls(learning_rate=rate, epochs=epochs).fit(X_fit, y_fit)
            scores = classification_metrics(y_val, predict_classes(model, X_val))
            rows.append({"model": name, "learning_rate": rate, "epochs": epochs,
                         "train_loss": model.loss_history_[-1], **scores})
            # Öncelik sınıf 1 F1; eşitlikte doğruluk, az epoch, küçük α.
            key = (scores["f1"], scores["accuracy"], -epochs, -rate)
            if best_key is None or key > best_key:
                best_key = key
                best = {"learning_rate": rate, "epochs": epochs,
                        "validation_f1": scores["f1"], "validation_accuracy": scores["accuracy"]}
    return best, rows


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = ROOT / "results"
    graphs = results / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    df, info = load_dataset(ROOT / "data/two_moons.csv")
    X, y = df[FEATURE_NAMES].to_numpy(dtype=float), encode_labels(df["label"])
    print(df.head().to_string(index=False))
    print(f"Boyut: {df.shape}; sütunlar: {list(df.columns)}")
    print("Sınıf dağılımı:", info["class_distribution"])
    print("Eksik değerler:", info["missing_values"])
    print("Yinelenen satır:", info["duplicate_rows"], "; IQR bayraklı satır:", info["iqr_flagged_rows"])
    df.describe().T.to_csv(results / "descriptive_statistics.csv", encoding="utf-8-sig")

    train, test = stratified_split(y, seed=SEED)
    inner_fit, inner_val = stratified_split(y[train], seed=SEED)
    fitting, validation = train[inner_fit], train[inner_val]
    split = np.full(len(y), "test", dtype=object)
    split[fitting], split[validation] = "inner_train", "validation"
    pd.DataFrame({"source_row": np.arange(1, len(y) + 1), "label": y, "split": split}).to_csv(
        results / "split_indices.csv", index=False, encoding="utf-8-sig")
    split_summary = {}
    for name, indices in [("outer_train", train), ("inner_train", fitting), ("validation", validation), ("test", test)]:
        split_summary[name] = {"n": len(indices), "class_0": int(np.sum(y[indices] == 0)),
                               "class_1": int(np.sum(y[indices] == 1))}
    info["split_summary"] = split_summary
    info["geometry_check"] = linear_nonseparability_certificate(X[train], y[train], train + 1)
    save_json(results / "dataset_info.json", info)
    save_json(results / "runtime_versions.json", {"python": sys.version, "numpy": version("numpy"),
                                                "pandas": version("pandas"), "matplotlib": version("matplotlib")})

    selected, candidates = {}, []
    # Dört deneyin bütün parametre seçimleri test tahmininden önce tamamlanır.
    for name, cls, polynomial in EXPERIMENTS:
        scaler = StandardScalerScratch()
        X_fit = scaler.fit_transform(feature_transform(X[fitting], polynomial))
        X_val = scaler.transform(feature_transform(X[validation], polynomial))
        selected[name], table = select_configuration(cls, X_fit, y[fitting], X_val, y[validation], name)
        candidates.extend(table)
        print(f"Doğrulama seçimi — {name}: {selected[name]}")
    save_json(results / "selected_hyperparameters.json", selected)
    pd.DataFrame(candidates).to_csv(results / "validation_results.csv", index=False, encoding="utf-8-sig")

    rows, pipelines, predictions, arrays = [], {}, {}, {}
    prediction_table = pd.DataFrame({"source_row": test + 1, "y_true": y[test]})
    for i, (name, cls, polynomial) in enumerate(EXPERIMENTS, start=1):
        settings = selected[name]
        # Son modelde ölçekler 240 dış eğitim örneğinden yeniden öğrenilir.
        scaler = StandardScalerScratch()
        X_train = scaler.fit_transform(feature_transform(X[train], polynomial))
        X_test = scaler.transform(feature_transform(X[test], polynomial))
        model = cls(learning_rate=settings["learning_rate"], epochs=settings["epochs"]).fit(X_train, y[train])
        raw = model.predict(X_test) if cls is LinearRegressionScratch else model.predict_proba(X_test)
        predicted = predict_classes(model, X_test)
        pipelines[name] = {"model": model, "scaler": scaler, "polynomial": polynomial}
        predictions[name] = predicted
        _, dw, db = model.loss_and_gradients(X_train, y[train], model.weights_, model.bias_)
        rows.append({"model": name, "feature_count": X_train.shape[1], **settings,
                     "train_loss": model.loss_history_[-1], "final_gradient_l2": float(np.linalg.norm(np.r_[dw, db])),
                     **classification_metrics(y[test], predicted)})
        prediction_table[f"{name} / raw"] = raw
        prediction_table[f"{name} / class"] = predicted
        names = POLYNOMIAL_NAMES if polynomial else FEATURE_NAMES
        pd.DataFrame({"feature": names, "mean": scaler.mean_, "std": scaler.scale_, "theta": model.weights_}).to_csv(
            results / f"parameters_{i}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"step": np.arange(len(model.loss_history_)), "training_loss": model.loss_history_}).to_csv(
            results / f"loss_history_{i}.csv", index=False)
        for key, value in [("mean", scaler.mean_), ("std", scaler.scale_), ("theta", model.weights_), ("bias", np.array(model.bias_))]:
            arrays[f"{key}_{i}"] = value

    # Eğitimde eşitlik varsa küçük etiket 0 seçilir; test sonucu kullanılmaz.
    majority = int(np.argmax(np.bincount(y[train], minlength=2)))
    baseline = np.full(len(test), majority)
    rows.append({"model": "Çoğunluk referansı", "feature_count": 0, "epochs": 0,
                 **classification_metrics(y[test], baseline)})
    prediction_table["Çoğunluk referansı / class"] = baseline
    prediction_table.to_csv(results / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(rows).set_index("model")
    metrics.to_csv(results / "metrics.csv", encoding="utf-8-sig")
    np.savez(results / "model_parameters.npz", train_indices=train, test_indices=test,
             inner_train_indices=fitting, validation_indices=validation, **arrays)
    plot_dataset(df, graphs)
    plot_results(pipelines, metrics, y[test], predictions, graphs)
    plot_boundaries(pipelines, X, y, train, test, graphs)
    write_summary(results, info, metrics, selected, pipelines, LEARNING_RATES, EPOCHS, majority)
    print("\nAynı test örneklerindeki gerçek sonuçlar:")
    print(metrics[["accuracy", "precision", "recall", "f1", "specificity"]].round(6).to_string())
    print(f"\nSonuçlar: {results}")


if __name__ == "__main__":
    main()
