"""NumPy tabanlı manuel veri hazırlama; testte fit yapılmaz."""

import hashlib

import numpy as np
import pandas as pd


FEATURE_NAMES = ["x1", "x2"]
POLYNOMIAL_NAMES = ["x1", "x2", "x1^2", "x1*x2", "x2^2"]


def encode_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Etiketler boş olmayan bir 0/1 vektörü olmalı.")
    return labels.astype(int)


def load_dataset(path):
    df = pd.read_csv(path)
    if list(df.columns) != ["x1", "x2", "label"]:
        raise ValueError(f"Beklenen sütunlar x1,x2,label; bulunan {list(df.columns)}.")
    if len(df) < 100 or df.isna().any().any():
        raise ValueError("En az 100 örnek ve eksiksiz veri gerekli; otomatik doldurma yapılmaz.")
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in df):
        raise ValueError("Tüm sütunlar sayısal olmalı.")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Sonlu olmayan veri bulundu.")
    y = encode_labels(df["label"])
    if set(y) != {0, 1}:
        raise ValueError("Tam olarak iki sınıf (0 ve 1) gerekli.")
    q1, q3 = df[FEATURE_NAMES].quantile(0.25), df[FEATURE_NAMES].quantile(0.75)
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flags = (df[FEATURE_NAMES] < low) | (df[FEATURE_NAMES] > high)
    info = {
        "file": "data/two_moons.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(df), "features": 2, "columns": list(df.columns),
        "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.duplicated(subset=FEATURE_NAMES).sum()),
        "ranges": {name: {"min": float(df[name].min()), "max": float(df[name].max())} for name in FEATURE_NAMES},
        "iqr_outliers": {name: {"lower": float(low[name]), "upper": float(high[name]),
                                "count": int(flags[name].sum())} for name in FEATURE_NAMES},
        "iqr_flagged_rows": int(flags.any(axis=1).sum()),
        "outlier_action": "Satır silinmedi, kırpılmadı; IQR bayrakları yalnızca betimleyicidir.",
        "original_source": "CSV içinde kaynak/üretim tohumu/gürültü bilgisi yok; özgün kaynak doğrulanamadı.",
    }
    return df, info


def stratified_split(y, test_size=0.2, seed=42):
    y = encode_labels(y)
    if not np.isscalar(test_size) or not 0 < test_size < 1:
        raise ValueError("Bölme oranı (0,1) içinde olmalı.")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(len(indices) * test_size))
        if n_test < 1 or n_test >= len(indices):
            raise ValueError("Her sınıfın her iki kümede bulunması için örnek sayısı yetersiz.")
        test.append(indices[:n_test])
        train.append(indices[n_test:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(test))


class StandardScalerScratch:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("Ölçekleme için sonlu, boş olmayan matris gerekli.")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.scale_ = np.where(std == 0, 1., std)
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Önce eğitim kümesinde fit çağrılmalı.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.mean_) or not np.isfinite(X).all():
            raise ValueError("Ölçeklenecek girdinin boyutu/değerleri geçersiz.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def polynomial_features(X):
    """İstenen sıra: [x1, x2, x1², x1*x2, x2²]; sabit sütun yoktur."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("Polinom dönüşümü iki sonlu özellik gerektirir.")
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1, x2, x1 ** 2, x1 * x2, x2 ** 2])


def feature_transform(X, polynomial=False):
    return polynomial_features(X) if polynomial else np.asarray(X, dtype=float).copy()
