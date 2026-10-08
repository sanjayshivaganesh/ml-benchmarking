"""Tests for confusion matrices, hard examples, and feature slices."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.analysis.error_analysis import analyze_all, analyze_errors
from src.models.models import MODEL_NAMES
from src.training.train import train_all


class _FixedClassifier:
    def __init__(self, predictions, probabilities=None, classes=None):
        self._predictions = np.asarray(predictions)
        self._probabilities = None if probabilities is None else np.asarray(probabilities, dtype=float)
        self.classes_ = None if classes is None else np.asarray(classes)
        self.fit_calls = 0
        if self._probabilities is None:
            self.predict_proba = None

    def fit(self, features, target):
        self.fit_calls += 1
        raise AssertionError("error analysis must not fit the model")

    def predict(self, features):
        return self._predictions.copy()

    def predict_proba(self, features):
        return self._probabilities.copy()


def _manual_example():
    features = pd.DataFrame(
        {
            "measurement": [0.1, 0.2, 0.4, 0.8, 1.1, 1.4],
            "group": ["a", "a", "b", "b", "a", "b"],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([1, 0, 0, 0, 0, 1])
    probabilities = np.array(
        [
            [0.2, 0.8],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.3, 0.7],
        ]
    )
    model = _FixedClassifier(predictions, probabilities, classes=[0, 1])
    return features, labels, model


class AnalyzeErrorsTests(unittest.TestCase):
    def test_confusion_matrix_dimensions_and_orientation(self):
        features, labels, model = _manual_example()
        result = analyze_errors(model, features, labels)
        matrix = result["confusion_matrix"]
        self.assertEqual(result["confusion_matrix_labels"], [0, 1])
        self.assertEqual(len(matrix), 2)
        self.assertTrue(all(len(row) == 2 for row in matrix))
        self.assertEqual(matrix, [[2, 1], [2, 1]])

    def test_false_positive_and_false_negative_counts(self):
        features, labels, model = _manual_example()
        result = analyze_errors(model, features, labels)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 2)
        kinds = [sample["error_type"] for sample in result["misclassified_samples"]]
        self.assertEqual(kinds.count("false_positive"), 1)
        self.assertEqual(kinds.count("false_negative"), 2)

    def test_error_counts_match_misclassified_samples(self):
        features, labels, model = _manual_example()
        result = analyze_errors(model, features, labels)
        self.assertEqual(result["misclassified_count"], 3)
        self.assertEqual(
            result["misclassified_count"],
            result["false_positives"] + result["false_negatives"],
        )
        self.assertEqual(result["misclassified_count"], len(result["misclassified_samples"]))
        self.assertAlmostEqual(result["misclassification_rate"], 0.5, places=12)
        self.assertEqual(result["n_test"], 6)

    def test_error_identification_is_deterministic(self):
        features, labels, model = _manual_example()
        labels = pd.Series(labels, index=features.index)
        before_features = features.copy()
        before_labels = labels.copy()
        first = analyze_errors(model, features, labels)
        second = analyze_errors(model, features, labels)
        expected_indexes = [10, 13, 14]
        self.assertEqual(
            [sample["sample_index"] for sample in first["misclassified_samples"]],
            expected_indexes,
        )
        self.assertEqual(first, second)
        self.assertEqual(model.fit_calls, 0)
        pd.testing.assert_frame_equal(features, before_features)
        pd.testing.assert_series_equal(labels, before_labels)
        false_positive = first["misclassified_samples"][0]
        self.assertEqual(false_positive["features"]["measurement"], 0.1)
        self.assertEqual(false_positive["features"]["group"], "a")
        self.assertAlmostEqual(false_positive["predicted_probability"], 0.8, places=12)
        self.assertAlmostEqual(false_positive["positive_class_probability"], 0.8, places=12)

    def test_results_are_json_serializable(self):
        features, labels, model = _manual_example()
        result = analyze_errors(model, features, labels)
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded), result)

    def test_missing_predict_proba_leaves_probabilities_empty(self):
        features, labels, _model = _manual_example()
        model = _FixedClassifier(predictions=np.array([1, 0, 0, 0, 0, 1]))
        result = analyze_errors(model, features, labels)
        for sample in result["misclassified_samples"]:
            self.assertIsNone(sample["predicted_probability"])
            self.assertIsNone(sample["positive_class_probability"])

    def test_numeric_quantile_bins_and_categorical_rates(self):
        measurement = np.arange(40)
        group = np.array(["low"] * 20 + ["high"] * 20)
        features = pd.DataFrame({"measurement": measurement, "group": group})
        labels = np.array([0] * 20 + [1] * 20)
        predictions = np.array([0] * 30 + [1] * 10)
        model = _FixedClassifier(predictions)
        result = analyze_errors(model, features, labels)
        self.assertEqual(set(result["feature_slices"]), {"measurement", "group"})

        numeric_bins = result["feature_slices"]["measurement"]
        self.assertEqual(numeric_bins["type"], "numeric")
        self.assertEqual([item["count"] for item in numeric_bins["bins"]], [10, 10, 10, 10])
        self.assertEqual([item["errors"] for item in numeric_bins["bins"]], [0, 0, 10, 0])

        categories = {
            item["category"]: item
            for item in result["feature_slices"]["group"]["categories"]
        }
        self.assertEqual(result["feature_slices"]["group"]["type"], "categorical")
        self.assertEqual(categories["low"]["error_rate"], 0.0)
        self.assertEqual(categories["high"]["count"], 20)
        self.assertEqual(categories["high"]["errors"], 10)

    def test_tiny_categories_are_omitted(self):
        features = pd.DataFrame(
            {
                "measurement": np.arange(200),
                "group": ["common"] * 180 + ["mid"] * 15 + ["rare"] * 5,
            }
        )
        labels = np.zeros(200, dtype=int)
        predictions = np.zeros(200, dtype=int)
        predictions[:10] = 1
        model = _FixedClassifier(predictions)
        result = analyze_errors(model, features, labels)
        categorical = result["feature_slices"]["group"]
        reported = {item["category"] for item in categorical["categories"]}
        self.assertEqual(reported, {"common"})
        self.assertEqual(categorical["omitted_small_categories"], 2)
        self.assertGreaterEqual(categorical["min_count"], 20)


class AnalyzeAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model_dir = root / "models"
        cls.report_path = root / "error_report.json"
        train_all(model_dir=cls.model_dir, random_state=42)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_models_are_analyzed_without_refitting(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("refit")):
            results = analyze_all(
                model_dir=self.model_dir,
                report_path=self.report_path,
                random_state=42,
            )
        self.assertEqual(list(results), ["breast_cancer", "adult"])
        for dataset_name, models in results.items():
            self.assertEqual(list(models), list(MODEL_NAMES))
            for report in models.values():
                matrix = report["confusion_matrix"]
                self.assertEqual(len(matrix), 2)
                self.assertTrue(all(len(row) == 2 for row in matrix))
                self.assertEqual(
                    report["misclassified_count"],
                    report["false_positives"] + report["false_negatives"],
                )
                self.assertEqual(
                    report["misclassified_count"],
                    len(report["misclassified_samples"]),
                )
                self.assertGreaterEqual(len(report["feature_slices"]), 1)
                self.assertLessEqual(len(report["feature_slices"]), 2)

        breast_slices = results["breast_cancer"]["logistic_regression"]["feature_slices"]
        self.assertTrue(all(item["type"] == "numeric" for item in breast_slices.values()))
        adult_slices = results["adult"]["logistic_regression"]["feature_slices"]
        self.assertEqual(
            {item["type"] for item in adult_slices.values()},
            {"numeric", "categorical"},
        )
        adult_sample = results["adult"]["logistic_regression"]["misclassified_samples"][0]
        self.assertLessEqual(len(adult_sample["features"]), 14)
        self.assertFalse(any("workclass_" in name for name in adult_sample["features"]))

        parsed = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, results)


if __name__ == "__main__":
    unittest.main()
