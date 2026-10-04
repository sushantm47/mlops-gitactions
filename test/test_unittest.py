"""Unittest suite for src/metrics.py."""

import unittest

from src import metrics
from src.evaluate import load_predictions


class TestClassificationMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        self.y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

    def test_accuracy(self):
        self.assertAlmostEqual(metrics.accuracy(self.y_true, self.y_pred), 0.75)

    def test_precision_recall_f1(self):
        cases = {
            metrics.precision: 0.75,
            metrics.recall: 0.75,
            metrics.f1_score: 0.75,
        }
        for func, expected in cases.items():
            with self.subTest(func=func.__name__):
                self.assertAlmostEqual(func(self.y_true, self.y_pred), expected)

    def test_confusion_matrix(self):
        self.assertEqual(
            metrics.confusion_matrix(self.y_true, self.y_pred),
            {"tp": 3, "fp": 1, "tn": 3, "fn": 1},
        )

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.accuracy([1, 0, 1], [1, 0])

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            metrics.accuracy("101", "101")


class TestRegressionMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [3.0, -0.5, 2.0, 7.0]
        self.y_pred = [2.5, 0.0, 2.0, 8.0]

    def test_mae(self):
        self.assertAlmostEqual(metrics.mae(self.y_true, self.y_pred), 0.5)

    def test_mse_and_rmse(self):
        self.assertAlmostEqual(metrics.mse(self.y_true, self.y_pred), 0.375)
        self.assertAlmostEqual(metrics.rmse(self.y_true, self.y_pred), 0.375 ** 0.5)

    def test_r2(self):
        self.assertAlmostEqual(metrics.r2_score(self.y_true, self.y_pred), 0.948608, places=5)

    def test_r2_zero_variance(self):
        with self.assertRaises(ValueError):
            metrics.r2_score([1, 1], [1, 2])


class TestQualityGate(unittest.TestCase):

    def test_saved_predictions_pass(self):
        y_true, y_pred = load_predictions()
        result = metrics.quality_gate(y_true, y_pred)
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["accuracy"], 0.8)

    def test_strict_threshold_fails(self):
        y_true, y_pred = load_predictions()
        self.assertFalse(metrics.quality_gate(y_true, y_pred, min_accuracy=0.99)["passed"])


if __name__ == "__main__":
    unittest.main()
