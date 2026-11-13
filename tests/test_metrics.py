import unittest

import numpy as np

from utils.metrics import classification_metrics, confusion_matrix


class MetricsTests(unittest.TestCase):
    def test_confusion_and_all_metrics(self):
        y, pred = [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1, 1]
        np.testing.assert_array_equal(confusion_matrix(y, pred), [[2, 1], [2, 3]])
        actual = classification_metrics(y, pred)
        for key, expected in {"accuracy": 5 / 8, "precision": 3 / 4, "recall": 3 / 5,
                               "f1": 2 / 3, "specificity": 2 / 3}.items():
            self.assertAlmostEqual(actual[key], expected)

    def test_zero_denominators(self):
        scores = classification_metrics([0, 0], [0, 0])
        self.assertEqual(scores["accuracy"], 1.)
        self.assertEqual(scores["precision"], 0.)
        self.assertEqual(scores["recall"], 0.)
        self.assertEqual(scores["f1"], 0.)
        self.assertEqual(classification_metrics([1], [1])["specificity"], 0.)

    def test_invalid_metrics_input(self):
        for y, pred in [([], []), ([0, 1], [1]), ([0, 2], [0, 1])]:
            with self.assertRaises(ValueError):
                classification_metrics(y, pred)
