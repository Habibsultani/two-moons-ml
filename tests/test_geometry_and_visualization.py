import unittest

import numpy as np

from models import LinearRegressionScratch
from utils.geometry import linear_nonseparability_certificate
from utils.preprocessing import StandardScalerScratch
from utils.visualization import decision_scores


class GeometryAndGridTests(unittest.TestCase):
    def test_crossing_segments_certificate(self):
        X, y = np.array([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]]), np.array([0, 0, 1, 1])
        certificate = linear_nonseparability_certificate(X, y, np.arange(1, 5))
        self.assertTrue(certificate["found"])
        np.testing.assert_allclose(certificate["intersection"], [0., 0.])
        self.assertLess(certificate["maximum_coordinate_error"], 1e-12)

    def test_separated_segments_no_certificate(self):
        X, y = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]]), np.array([0, 0, 1, 1])
        self.assertFalse(linear_nonseparability_certificate(X, y, np.arange(1, 5))["found"])

    def test_prediction_grid_uses_training_scaler(self):
        scaler = StandardScalerScratch().fit([[0., 0.], [2., 4.]])
        model = LinearRegressionScratch()
        model.weights_, model.bias_ = np.array([0.25, 0.5]), 0.1
        pipeline = {"model": model, "scaler": scaler, "polynomial": False}
        np.testing.assert_allclose(decision_scores(pipeline, np.array([[3., 6.]])), [1.6])
        np.testing.assert_allclose(scaler.mean_, [1., 2.])
