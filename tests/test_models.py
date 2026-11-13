import unittest

import numpy as np

from models import LinearRegressionScratch, LogisticRegressionScratch


class ModelTests(unittest.TestCase):
    def test_linear_prediction_and_threshold(self):
        model = LinearRegressionScratch()
        model.weights_, model.bias_ = np.array([2., -1.]), 0.5
        X = np.array([[0., 0.], [1., 0.], [-1., 0.]])
        np.testing.assert_allclose(model.predict(X), [0.5, 2.5, -1.5])
        np.testing.assert_array_equal(model.predict_classes(X), [1, 1, 0])

    def test_linear_mse_convention(self):
        loss, dw, db = LinearRegressionScratch.loss_and_gradients(
            np.array([[1.], [2.]]), np.array([0., 1.]), np.array([0.]), 0.)
        self.assertEqual(loss, 0.25)
        np.testing.assert_allclose(dw, [-1.])
        self.assertEqual(db, -0.5)

    def test_sigmoid_extremes_and_scalar(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values = LogisticRegressionScratch.sigmoid([-1000., -1., 0., 1., 1000.])
        np.testing.assert_allclose(values, [0., 1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-1)), 1.])
        self.assertEqual(float(LogisticRegressionScratch.sigmoid(0.)), 0.5)

    def test_logistic_bce_and_extreme_incorrect_predictions(self):
        X, y = np.array([[-1.], [1.]]), np.array([1., 0.])
        loss, _, _ = LogisticRegressionScratch.loss_and_gradients(X, y, np.array([0.]), 0.)
        self.assertAlmostEqual(loss, np.log(2))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss, dw, db = LogisticRegressionScratch.loss_and_gradients(X, y, np.array([1000.]), 0.)
        self.assertEqual(loss, 1000.)
        self.assertTrue(np.isfinite(dw).all() and np.isfinite(db))

    def test_gradients_with_independent_finite_differences(self):
        rng = np.random.default_rng(123)
        for cls in [LinearRegressionScratch, LogisticRegressionScratch]:
            for m, n in [(1, 2), (11, 2), (13, 5)]:
                with self.subTest(model=cls.__name__, m=m, n=n):
                    X, y = rng.normal(size=(m, n)), rng.integers(0, 2, size=m)
                    params = rng.normal(scale=0.3, size=n + 1)
                    def independent_loss(p):
                        z = X @ p[:-1] + p[-1]
                        if cls is LinearRegressionScratch:
                            return np.sum((z - y) ** 2) / (2 * m)
                        probability = 1 / (1 + np.exp(-z))
                        return -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))
                    loss, dw, db = cls.loss_and_gradients(X, y, params[:-1], params[-1])
                    numerical = np.zeros(n + 1)
                    for j in range(n + 1):
                        offset = np.zeros(n + 1)
                        offset[j] = 1e-6
                        numerical[j] = (independent_loss(params + offset) - independent_loss(params - offset)) / 2e-6
                    self.assertAlmostEqual(loss, independent_loss(params))
                    np.testing.assert_allclose(np.r_[dw, db], numerical, atol=1e-8, rtol=1e-6)

    def test_one_batch_update_includes_bias_and_learning_rate(self):
        X, y = np.array([[1., 2.], [3., -1.]]), np.array([0., 1.])
        for cls in [LinearRegressionScratch, LogisticRegressionScratch]:
            _, dw, db = cls.loss_and_gradients(X, y, np.zeros(2), 0.)
            model = cls(learning_rate=0.1, epochs=1).fit(X, y)
            np.testing.assert_allclose(model.weights_, -0.1 * dw)
            self.assertAlmostEqual(model.bias_, -0.1 * db)
            self.assertEqual(len(model.loss_history_), 2)
            self.assertAlmostEqual(model.loss_history_[-1], cls.loss_and_gradients(X, y, model.weights_, model.bias_)[0])

    def test_linear_training_on_known_relationship(self):
        X = np.array([[-2.], [-1.], [0.], [1.], [2.]])
        target = 2 * X[:, 0] + 1
        model = LinearRegressionScratch(0.1, 400).fit(X, target)
        np.testing.assert_allclose(model.predict(X), target, atol=1e-8)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_logistic_training_and_reset(self):
        X, y = np.array([[-2.], [-1.], [1.], [2.]]), np.array([0, 0, 1, 1])
        model = LogisticRegressionScratch(0.1, 400).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertTrue(((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1)).all())
        old = model.weights_.copy()
        model.fit(X, y)
        np.testing.assert_array_equal(model.weights_, old)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_unfitted_invalid_shapes_and_parameters(self):
        for cls in [LinearRegressionScratch, LogisticRegressionScratch]:
            with self.assertRaises(RuntimeError):
                cls().predict([[1.]])
            for rate, epochs in [(-1., 10), (np.nan, 10), (0.1, 0), (0.1, True)]:
                with self.assertRaises(ValueError):
                    cls(rate, epochs).fit([[1.], [2.]], [0, 1])
            for X, y in [([[1.], [2.]], [[0], [1]]), ([[np.nan], [1.]], [0, 1]), ([[1.]], [0, 1])]:
                with self.assertRaises(ValueError):
                    cls().fit(X, y)
        with self.assertRaises(ValueError):
            LogisticRegressionScratch().fit([[1.], [2.]], [0, 2])

    def test_logistic_half_threshold_and_invalid_threshold(self):
        model = LogisticRegressionScratch()
        model.weights_, model.bias_ = np.zeros(1), 0.
        np.testing.assert_array_equal(model.predict([[0.], [1.]]), [1, 1])
        for threshold in [np.nan, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                model.predict([[0.]], threshold)
