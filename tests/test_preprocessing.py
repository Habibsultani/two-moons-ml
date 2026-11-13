from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.preprocessing import (StandardScalerScratch, encode_labels, load_dataset,
                                 polynomial_features, stratified_split)


class PreprocessingTests(unittest.TestCase):
    def test_labels(self):
        np.testing.assert_array_equal(encode_labels([1., 0., 1.]), [1, 0, 1])
        for values in [[], [0, 2], ["NO", "AB"]]:
            with self.assertRaises(ValueError):
                encode_labels(values)

    def test_stratification_coverage_reproducibility(self):
        y = np.r_[np.zeros(150, dtype=int), np.ones(150, dtype=int)]
        train, test = stratified_split(y)
        self.assertEqual((len(train), len(test)), (240, 60))
        np.testing.assert_array_equal(np.bincount(y[train]), [120, 120])
        np.testing.assert_array_equal(np.bincount(y[test]), [30, 30])
        np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(300))
        self.assertEqual(len(np.intersect1d(train, test)), 0)
        np.testing.assert_array_equal(test, stratified_split(y)[1])
        self.assertFalse(np.array_equal(test, stratified_split(y, seed=43)[1]))
        fitting, val = stratified_split(y[train])
        self.assertEqual((len(fitting), len(val)), (192, 48))
        np.testing.assert_array_equal(np.bincount(y[train][val]), [24, 24])

    def test_training_only_standardization_and_constant_column(self):
        training, held_out = np.array([[1., 7.], [3., 7.]]), np.array([[101., 7.]])
        scaler = StandardScalerScratch().fit(training)
        np.testing.assert_allclose(scaler.mean_, [2., 7.])
        np.testing.assert_allclose(scaler.scale_, [1., 1.])
        np.testing.assert_allclose(scaler.transform(training), [[-1., 0.], [1., 0.]])
        np.testing.assert_allclose(scaler.transform(held_out), [[99., 0.]])
        np.testing.assert_allclose(scaler.mean_, [2., 7.])
        np.testing.assert_array_equal(training, [[1., 7.], [3., 7.]])

    def test_polynomial_order_and_no_fit_on_held_out(self):
        np.testing.assert_array_equal(polynomial_features([[2., 3.], [-1., 2.]]),
                                      [[2., 3., 4., 6., 9.], [-1., 2., 1., -2., 4.]])
        train = polynomial_features([[1., 2.], [2., 4.], [3., 3.]])
        scaler = StandardScalerScratch().fit(train)
        before = scaler.mean_.copy()
        scaler.transform(polynomial_features([[100., 200.]]))
        np.testing.assert_allclose(scaler.mean_, before)

    def test_invalid_split_scaler_and_polynomial(self):
        with self.assertRaises(ValueError):
            stratified_split([0, 1])
        with self.assertRaises(RuntimeError):
            StandardScalerScratch().transform([[1.]])
        with self.assertRaises(ValueError):
            polynomial_features([[1., 2., 3.]])
        with self.assertRaises(ValueError):
            StandardScalerScratch().fit([[np.inf]])

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            frame = pd.DataFrame({"x1": np.arange(100), "x2": np.arange(100) / 2, "label": np.arange(100) % 2})
            frame.to_csv(path, index=False)
            actual, info = load_dataset(path)
            self.assertEqual(actual.shape, (100, 3))
            self.assertEqual(info["class_distribution"], {"0": 50, "1": 50})
            frame.loc[0, "x1"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(path)
