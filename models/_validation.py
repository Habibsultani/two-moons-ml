"""Modellerin ortak boyut ve sayısal girdi denetimleri."""

import numpy as np


def training_data(X, y, learning_rate, epochs):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X boş olmayan iki boyutlu bir matris olmalı.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y, X ile aynı satır sayısında tek boyutlu olmalı.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Veriler sonlu sayılar içermeli.")
    if not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("α pozitif ve sonlu olmalı.")
    if isinstance(epochs, (bool, np.bool_)) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("Epoch pozitif tam sayı olmalı.")
    return X, y


def prediction_data(X, weights):
    if weights is None:
        raise RuntimeError("Tahminden önce fit çağrılmalı.")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(weights) or not np.isfinite(X).all():
        raise ValueError("Tahmin girdisinin boyutu/değerleri geçersiz.")
    return X


def check_threshold(threshold):
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("Eşik sonlu bir sayı olmalı.")
