"""Tests for subgroup performance slicing on existing predictions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.slicing import (
    MIN_SLICE_SIZE,
    run_error_slicing,
    save_experiment_slices,
    save_slicing_result,
)
from src.evaluation.evaluate import score_predictions


class NumericalSliceTests(unittest.TestCase):
    def test_quantile_bins_are_labeled_in_order(self):
        features = pd.DataFrame({"age": np.arange(40)})
        labels = np.zeros(40, dtype=int)
        result = run_error_slicing(
            features,
            labels,
            labels,
            n_bins=4,
            min_slice_size=1,
        )
        definitions = [item["slice_definition"] for item in result.slices]
        self.assertEqual([item.split(":", 1)[0] for item in definitions], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual([item["sample_count"] for item in result.slices], [10, 10, 10, 10])
        self.assertTrue(all(item["feature_type"] == "numeric" for item in result.slices))

    def test_duplicate_edges_do_not_create_invalid_bins(self):
        features = pd.DataFrame({"age": [0] * 15 + [1] * 15})
        labels = np.zeros(30, dtype=int)
        result = run_error_slicing(
            features,
            labels,
            labels,
            n_bins=4,
            min_slice_size=1,
        )
        self.assertGreaterEqual(len(result.slices), 1)
        self.assertLessEqual(len(result.slices), 2)
        self.assertTrue(
            all(item["slice_definition"].startswith("Q") for item in result.slices)
        )

    def test_constant_numeric_feature_is_one_slice(self):
        features = pd.DataFrame({"age": [5] * 10})
        labels = np.zeros(10, dtype=int)
        result = run_error_slicing(
            features,
            labels,
            labels,
            n_bins=4,
            min_slice_size=1,
        )
        self.assertEqual(len(result.slices), 1)
        self.assertEqual(result.slices[0]["slice_definition"], "constant = 5")
        self.assertEqual(result.slices[0]["sample_count"], 10)

    def test_missing_numeric_values_are_a_separate_slice(self):
        features = pd.DataFrame({"age": list(range(20)) + [np.nan, np.nan, np.nan]})
        labels = np.zeros(23, dtype=int)
        result = run_error_slicing(
            features,
            labels,
            labels,
            n_bins=4,
            min_slice_size=1,
        )
        missing = [
            item for item in result.slices if item["slice_definition"] == "(missing)"
        ]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["sample_count"], 3)
        self.assertEqual(sum(item["sample_count"] for item in result.slices), 23)
        quantile_counts = [
            item["sample_count"]
            for item in result.slices
            if item["slice_definition"].startswith("Q")
        ]
        self.assertEqual(sum(quantile_counts), 20)


class CategoricalAndMetricTests(unittest.TestCase):
    def test_categories_group_directly_and_match_score_predictions(self):
        features = pd.DataFrame({"workclass": ["Private"] * 12 + ["Local-gov"] * 8})
        labels = np.array([1] * 12 + [0] * 8)
        predicted = np.array([1] * 10 + [0] * 2 + [0] * 8)
        result = run_error_slicing(
            features,
            labels,
            predicted,
            min_slice_size=1,
        )
        counts = {
            item["slice_definition"]: item["sample_count"] for item in result.slices
        }
        self.assertEqual(counts, {"Local-gov": 8, "Private": 12})
        private = next(item for item in result.slices if item["slice_definition"] == "Private")
        expected = score_predictions(labels[:12], predicted[:12])
        self.assertEqual(private["metrics"], expected)
        self.assertAlmostEqual(
            private["delta_f1"],
            private["metrics"]["f1"] - result.overall_metrics["f1"],
        )
        self.assertAlmostEqual(
            private["delta_accuracy"],
            private["metrics"]["accuracy"] - result.overall_metrics["accuracy"],
        )
        self.assertEqual(result.overall_metrics, score_predictions(labels, predicted))

    def test_small_slices_are_not_ranked_as_reliable(self):
        self.assertEqual(MIN_SLICE_SIZE, 20)
        features = pd.DataFrame({"group": ["kept"] * 20 + ["small"] * 5})
        labels = np.array([1] * 25)
        predicted = np.array([1] * 20 + [0] * 5)
        result = run_error_slicing(features, labels, predicted)
        small = next(item for item in result.slices if item["slice_definition"] == "small")
        self.assertFalse(small["reliable"])
        self.assertIsNone(small["delta_f1"])
        self.assertEqual(small["min_slice_size"], 20)
        self.assertEqual(
            [item["slice_definition"] for item in result.worst_slices],
            ["kept"],
        )

    def test_slices_without_class_1_are_not_ranked_as_failures(self):
        features = pd.DataFrame({"group": ["negative"] * 25 + ["mixed"] * 25})
        labels = np.array([0] * 25 + [1] * 25)
        predicted = np.array([0] * 25 + [0] * 10 + [1] * 15)
        result = run_error_slicing(features, labels, predicted, min_slice_size=20)
        negative = next(item for item in result.slices if item["slice_definition"] == "negative")
        self.assertTrue(negative["reliable"])
        self.assertFalse(negative["f1_comparable"])
        self.assertIsNone(negative["delta_f1"])
        self.assertEqual(negative["metrics"]["accuracy"], 1.0)
        self.assertEqual(negative["metrics"]["f1"], 0.0)
        self.assertEqual(
            [item["slice_definition"] for item in result.worst_slices],
            ["mixed"],
        )

    def test_no_reliable_slices_leave_the_worst_list_empty(self):
        features = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5})
        labels = np.zeros(10, dtype=int)
        result = run_error_slicing(features, labels, labels, min_slice_size=20)
        self.assertEqual(result.worst_slices, ())
        self.assertTrue(result.slices)
        self.assertTrue(all(not item["reliable"] for item in result.slices))
        self.assertTrue(result.table["reliable"].eq(False).all())

    def test_worst_slices_are_ranked_by_f1_degradation(self):
        features = pd.DataFrame(
            {"group": ["bad"] * 20 + ["good"] * 20 + ["tiny"] * 5}
        )
        labels = np.ones(45, dtype=int)
        predicted = np.array([0] * 20 + [1] * 20 + [0] * 5)
        result = run_error_slicing(
            features,
            labels,
            predicted,
            dataset="adult",
            model="logistic_regression",
            min_slice_size=20,
        )
        self.assertEqual(
            [item["slice_definition"] for item in result.worst_slices],
            ["bad", "good"],
        )
        self.assertLess(result.worst_slices[0]["delta_f1"], 0)
        self.assertGreater(result.worst_slices[1]["delta_f1"], 0)
        self.assertEqual(result.table.loc[result.table["reliable"], "worst_rank"].tolist(), [1, 2])
        payload = result.to_dict()
        self.assertTrue(payload["association_only"])
        self.assertIn("do not show that a feature caused", payload["note"])

    def test_slice_json_records_reliability_and_deltas(self):
        features = pd.DataFrame({"group": ["bad"] * 20 + ["good"] * 20})
        labels = np.ones(40, dtype=int)
        predicted = np.array([0] * 20 + [1] * 20)
        result = run_error_slicing(
            features,
            labels,
            predicted,
            dataset="breast_cancer",
            model="logistic_regression",
            min_slice_size=20,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_slicing_result(result, Path(tmp) / "outputs" / "diagnostics" / "slicing")
            self.assertEqual(path.name, "breast_cancer_logistic_regression_slices.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"], "breast_cancer")
        self.assertEqual(payload["model"], "logistic_regression")
        self.assertEqual(payload["min_slice_size"], 20)
        self.assertEqual(set(payload["overall_metrics"]), {"accuracy", "precision", "recall", "f1"})
        row = payload["slices"][0]
        self.assertIn("sample_count", row)
        self.assertIn("metrics", row)
        self.assertIn("delta_f1", row)
        self.assertIn("reliable", row)


class ExperimentSliceFileTests(unittest.TestCase):
    def test_experiment_json_uses_the_diagnostics_directory(self):
        y_test = pd.Series(np.array([0] * 20 + [1] * 20), index=np.arange(40))
        features = pd.DataFrame({"age": np.arange(40)}, index=y_test.index)
        model = type("Model", (), {})()
        model.predictions = np.array([0] * 20 + [1] * 20)
        result = type("Result", (), {})()
        result.dataset = "breast_cancer"
        result.model = None
        result.models = ("logistic_regression",)
        result.y_test = y_test
        result.X_test = features
        result.fitted = {"logistic_regression": model}
        result.metadata = {"numeric_features": ["age"], "categorical_features": []}
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "outputs" / "experiments" / "run"
            written = save_experiment_slices(result, model_dir)
            path = written["logistic_regression"]
            self.assertEqual(
                path,
                Path(tmp)
                / "outputs"
                / "diagnostics"
                / "slicing"
                / "breast_cancer_logistic_regression_slices.json",
            )
            self.assertTrue(path.is_file())
