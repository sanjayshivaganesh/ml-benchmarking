"""Tests for held-out classification metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.evaluate import METRIC_NAMES, evaluate_all, evaluate_model
from src.models.models import MODEL_NAMES
from src.training.train import train_all


class _FixedClassifier:
    """Deterministic stub with an explicit predict and predict_proba."""

    def __init__(self, predictions, probabilities, classes):
        self._predictions = np.asarray(predictions)
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes)
        self.fit_calls = 0

    def fit(self, features, target):
        self.fit_calls += 1
        raise AssertionError("evaluation must not fit the model")

    def predict(self, features):
        return self._predictions.copy()

    def predict_proba(self, features):
        return self._probabilities.copy()


class _LabelsOnlyClassifier:
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def decision_function(self, features):
        return np.zeros(len(features), dtype=float)


def _known_example():
    features = np.arange(4).reshape(-1, 1)
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])
    # Column 1 is P(y=1): [0.2, 0.7, 0.4, 0.9]
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    model = _FixedClassifier(predictions, probabilities, classes=[0, 1])
    return features, labels, model


class EvaluateModelTests(unittest.TestCase):
    def test_metric_keys(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        self.assertEqual(list(metrics), list(METRIC_NAMES))

    def test_metric_ranges_and_python_floats(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        for name in METRIC_NAMES:
            value = metrics[name]
            self.assertIs(type(value), float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_roc_auc_uses_positive_class_probabilities(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        # Pairs of (0.4, 0.9) against (0.2, 0.7): 3 of 4 are ranked correctly.
        self.assertAlmostEqual(metrics["roc_auc"], 0.75, places=12)

    def test_brier_score_uses_positive_class_probabilities(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        probabilities = np.array([0.2, 0.7, 0.4, 0.9])
        expected = float(np.mean((probabilities - labels) ** 2))
        self.assertAlmostEqual(expected, 0.225, places=12)
        self.assertAlmostEqual(metrics["brier_score"], expected, places=12)

    def test_label_metrics_use_predict_not_a_custom_threshold(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=12)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=12)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=12)
        self.assertAlmostEqual(metrics["f1"], 0.5, places=12)
        self.assertEqual(model.fit_calls, 0)

    def test_swapped_class_columns_are_not_silently_reversed(self):
        features = np.arange(4).reshape(-1, 1)
        labels = np.array([0, 0, 1, 1])
        # Column 0 is P(y=1) because classes_ is [1, 0].
        probabilities = np.array(
            [
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.9, 0.1],
            ]
        )
        model = _FixedClassifier(
            predictions=np.array([0, 1, 0, 1]),
            probabilities=probabilities,
            classes=[1, 0],
        )
        metrics = evaluate_model(model, features, labels)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75, places=12)
        self.assertAlmostEqual(metrics["brier_score"], 0.225, places=12)

    def test_zero_division_returns_zero_instead_of_raising(self):
        features = np.arange(2).reshape(-1, 1)
        labels = np.array([0, 1])
        model = _FixedClassifier(
            predictions=np.array([0, 0]),
            probabilities=np.array([[0.9, 0.1], [0.6, 0.4]]),
            classes=[0, 1],
        )
        metrics = evaluate_model(model, features, labels)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_missing_predict_proba_raises(self):
        features = np.arange(2).reshape(-1, 1)
        labels = np.array([0, 1])
        with self.assertRaises(TypeError):
            evaluate_model(_LabelsOnlyClassifier(), features, labels)

    def test_probabilities_outside_unit_interval_raise(self):
        features = np.arange(2).reshape(-1, 1)
        labels = np.array([0, 1])
        model = _FixedClassifier(
            predictions=np.array([0, 1]),
            probabilities=np.array([[0.2, -0.1], [0.4, 1.1]]),
            classes=[0, 1],
        )
        with self.assertRaises(ValueError):
            evaluate_model(model, features, labels)

    def test_repeated_calls_are_deterministic(self):
        features, labels, model = _known_example()
        labels = pd.Series(labels)
        before = labels.copy()
        first = evaluate_model(model, features, labels)
        second = evaluate_model(model, features, labels)
        self.assertEqual(first, second)
        pd.testing.assert_series_equal(labels, before)

    def test_metrics_are_json_serializable(self):
        features, labels, model = _known_example()
        metrics = evaluate_model(model, features, labels)
        encoded = json.dumps(metrics, allow_nan=False)
        parsed = json.loads(encoded)
        self.assertEqual(list(parsed), list(METRIC_NAMES))
        for name in METRIC_NAMES:
            self.assertIsInstance(parsed[name], float)
            self.assertAlmostEqual(parsed[name], metrics[name], places=12)


class EvaluateAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model_dir = root / "models"
        cls.metrics_path = root / "metrics.json"
        train_all(model_dir=cls.model_dir, random_state=42)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_saved_models_are_scored_without_refitting(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("refit")):
            results = evaluate_all(
                model_dir=self.model_dir,
                metrics_path=self.metrics_path,
                random_state=42,
            )
        self.assertEqual(list(results), ["breast_cancer", "adult"])
        for dataset_name, models in results.items():
            self.assertEqual(list(models), list(MODEL_NAMES))
            for model_name, metrics in models.items():
                self.assertEqual(list(metrics), list(METRIC_NAMES))
                for name, value in metrics.items():
                    self.assertIs(type(value), float)
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

        parsed = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, results)
        again = evaluate_all(
            model_dir=self.model_dir,
            metrics_path=self.metrics_path,
            random_state=42,
        )
        self.assertEqual(again, results)


if __name__ == "__main__":
    unittest.main()
