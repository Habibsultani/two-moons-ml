"""Kararlı sigmoid/BCE ve elle yazılmış batch Gradient Descent."""

import numpy as np

from ._validation import check_threshold, prediction_data, training_data


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=3000):
        self.learning_rate, self.epochs = learning_rate, epochs
        self.weights_, self.bias_ = None, 0.0
        self.loss_history_ = []

    @staticmethod
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1 + exp_z)
        return result

    @staticmethod
    def loss_and_gradients(X, y, theta, theta_0):
        z = X @ theta + theta_0
        # BCE'nin eşdeğer kararlı biçimi; log(0) ve exp taşmasını önler.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        residual = LogisticRegressionScratch.sigmoid(z) - y
        dtheta = X.T @ residual / len(y)
        dtheta_0 = float(np.mean(residual))
        return loss, dtheta, dtheta_0

    def fit(self, X, y):
        X, y = training_data(X, y, self.learning_rate, self.epochs)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Lojistik etiketler 0 veya 1 olmalı.")
        self.weights_, self.bias_ = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            loss, dtheta, dtheta_0 = self.loss_and_gradients(X, y, self.weights_, self.bias_)
            if not np.isfinite(loss):
                raise FloatingPointError("Lojistik kayıp sonlu değil.")
            self.loss_history_.append(loss)
            self.weights_ -= self.learning_rate * dtheta
            self.bias_ -= self.learning_rate * dtheta_0
        final_loss = self.loss_and_gradients(X, y, self.weights_, self.bias_)[0]
        if not np.isfinite(final_loss):
            raise FloatingPointError("Son eğitim kaybı sonlu değil.")
        self.loss_history_.append(final_loss)
        return self

    def predict_proba(self, X):
        """Sınıf 1 için model olasılığı; kalibrasyon ayrıca sınanmadı."""
        X = prediction_data(X, self.weights_)
        return self.sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X, threshold=0.5):
        check_threshold(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("Olasılık eşiği [0,1] aralığında olmalı.")
        return (self.predict_proba(X) >= threshold).astype(int)
