"""Pozitif sınıf 1; matris satırları gerçek, sütunları tahmin: [[TN,FP],[FN,TP]]."""

import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or not len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("Etiketler eşit uzunlukta, boş olmayan vektörler olmalı.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Etiketler 0 veya 1 olmalı.")
    return np.array([[np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
                     [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))]], dtype=int)


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    def divide(a, b):
        return float(a / b) if b else 0.0
    return {"accuracy": divide(tn + tp, tn + fp + fn + tp), "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn),
            "specificity": divide(tn, tn + fp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
