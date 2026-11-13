"""J(θ)=Σ(ŷ−y)²/(2m); batch Gradient Descent elle uygulanır."""

import numpy as np

from ._validation import check_threshold, prediction_data, training_data


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=3000):
        self.learning_rate, self.epochs = learning_rate, epochs
        self.weights_, self.bias_ = None, 0.0
        self.loss_history_ = []

    @staticmethod
    def loss_and_gradients(X, y, theta, theta_0):
        residual = X @ theta + theta_0 - y
        # 1/2 çarpanı türevdeki 2'yi götürür; her iki türev 1/m içerir.
        loss = float(np.mean(residual ** 2) / 2)
        dtheta = X.T @ residual / len(y)
        dtheta_0 = float(np.mean(residual))
        return loss, dtheta, dtheta_0

    def fit(self, X, y):
        X, y = training_data(X, y, self.learning_rate, self.epochs)
        self.weights_, self.bias_ = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            loss, dtheta, dtheta_0 = self.loss_and_gradients(X, y, self.weights_, self.bias_)
            if not np.isfinite(loss):
                raise FloatingPointError("Doğrusal kayıp sonlu değil; α değerini inceleyin.")
            self.loss_history_.append(loss)
            # θ ← θ − α∇J; θ₀ ← θ₀ − α∂J/∂θ₀
            self.weights_ -= self.learning_rate * dtheta
            self.bias_ -= self.learning_rate * dtheta_0
        final_loss = self.loss_and_gradients(X, y, self.weights_, self.bias_)[0]
        if not np.isfinite(final_loss):
            raise FloatingPointError("Son eğitim kaybı sonlu değil.")
        self.loss_history_.append(final_loss)
        return self

    def predict(self, X):
        """Ham regresyon çıktısı; olasılık değildir ve [0,1] dışında olabilir."""
        X = prediction_data(X, self.weights_)
        return X @ self.weights_ + self.bias_

    def predict_classes(self, X, threshold=0.5):
        check_threshold(threshold)
        return (self.predict(X) >= threshold).astype(int)
