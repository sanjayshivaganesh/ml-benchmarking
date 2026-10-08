"""Tests for confident mistakes and uncertain predictions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.hard_examples import (
    DEFAULT_TOP_N,
    explore_hard_examples,
    hard_examples_directory,
    save_hard_examples,
)
from src.diagnostics.prediction_store import PredictionStore


class HardExampleTests(unittest.TestCase):
    def test_confident_wrongs_rank_by_confidence(self):
        store = _store(
            [1, 2, 3, 4],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0.95, 0.40, 0.80, 0.90],
        )
        result = explore_hard_examples(store, top_n=2)
        ranked = [item["sample_id"] for item in result.high_confidence_errors]
        self.assertEqual(ranked, [1, 3])
        self.assertGreater(
            result.high_confidence_errors[0]["confidence"],
            result.high_confidence_errors[1]["confidence"],
        )
        self.assertTrue(all(item["correct"] is False for item in result.high_confidence_errors))

    def test_uncertainty_is_highest_at_one_half(self):
        store = _store(
            [10, 11, 12, 13],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0.5, 0.9, 0.1, 0.6],
        )
        result = explore_hard_examples(store, top_n=4)
        self.assertEqual(
            [item["sample_id"] for item in result.most_uncertain],
            [10, 13, 11, 12],
        )
        by_id = {item["sample_id"]: item["uncertainty"] for item in result.most_uncertain}
        self.assertEqual(by_id[10], 1.0)
        self.assertAlmostEqual(by_id[11], 0.2)
        self.assertAlmostEqual(by_id[12], 0.2)
        self.assertAlmostEqual(by_id[13], 0.8)
        probabilities = (0.5, 0.9, 0.1, 0.6)
        uncertainty = np.array([1.0 - 2.0 * abs(probability - 0.5) for probability in probabilities])
        self.assertEqual(result.summary["mean_uncertainty"], float(np.mean(uncertainty)))
        self.assertEqual(result.summary["n_samples"], 4)

    def test_binary_edges_and_four_categories(self):
        store = _store(
            [1, 2, 3, 4],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [0.9, 0.55, 0.95, 0.4],
        )
        result = explore_hard_examples(store, top_n=10)
        self.assertEqual(
            result.summary["correct_confident"]
            + result.summary["correct_uncertain"]
            + result.summary["wrong_confident"]
            + result.summary["wrong_uncertain"],
            4,
        )
        categories = {item["sample_id"]: item["category"] for item in result.most_uncertain}
        self.assertEqual(categories[1], "correct_confident")
        self.assertEqual(categories[2], "correct_uncertain")
        self.assertEqual(categories[3], "wrong_confident")
        self.assertEqual(categories[4], "wrong_uncertain")
        edges = explore_hard_examples(
            _store([1, 2, 3], [0, 1, 1], [0, 1, 1], [0.0, 0.5, 1.0]),
            top_n=3,
        )
        uncertainty = {item["sample_id"]: item["uncertainty"] for item in edges.most_uncertain}
        self.assertEqual(uncertainty[1], 0.0)
        self.assertEqual(uncertainty[2], 1.0)
        self.assertEqual(uncertainty[3], 0.0)

    def test_feature_values_are_kept_with_each_example(self):
        features = pd.DataFrame(
            {"age": [41, np.nan], "workclass": ["Private", None]},
            index=[7, 8],
        )
        store = _store([7, 8], [0, 1], [1, 1], [0.99, 0.5], features)
        result = explore_hard_examples(
            store,
            dataset="adult",
            model="logistic_regression",
            top_n=2,
        )
        mistake = result.high_confidence_errors[0]
        self.assertEqual(mistake["sample_id"], 7)
        self.assertEqual(mistake["features"]["age"], 41)
        self.assertEqual(mistake["features"]["workclass"], "Private")
        uncertain = next(item for item in result.most_uncertain if item["sample_id"] == 8)
        self.assertIsNone(uncertain["features"]["age"])
        self.assertIsNone(uncertain["features"]["workclass"])
        self.assertIn("y_true", mistake)
        self.assertIn("y_pred", mistake)
        self.assertIn("probability", mistake)

    def test_top_n_limits_both_lists(self):
        self.assertEqual(DEFAULT_TOP_N, 20)
        store = _store(
            list(range(6)),
            [0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0.99, 0.9, 0.8, 0.7, 0.6, 0.55],
        )
        result = explore_hard_examples(store, top_n=2)
        self.assertEqual(len(result.high_confidence_errors), 2)
        self.assertEqual(
            [item["sample_id"] for item in result.high_confidence_errors],
            [0, 1],
        )
        self.assertEqual(len(result.most_uncertain), 2)
        larger = explore_hard_examples(store, top_n=10)
        self.assertEqual(len(larger.high_confidence_errors), 4)

    def test_empty_error_set(self):
        store = _store([3, 1, 2], [0, 1, 1], [0, 1, 1], [0.1, 0.8, 0.5])
        result = explore_hard_examples(store, top_n=5)
        self.assertEqual(result.high_confidence_errors, ())
        self.assertEqual(result.summary["n_errors"], 0)
        self.assertEqual(result.summary["wrong_confident"], 0)
        self.assertEqual(result.summary["wrong_uncertain"], 0)
        self.assertEqual(len(result.most_uncertain), 3)

    def test_identical_predictions_stay_ordered_by_sample_id(self):
        store = _store([5, 1, 4], [1, 1, 1], [1, 1, 1], [0.8, 0.8, 0.8])
        result = explore_hard_examples(store, top_n=3)
        self.assertEqual(result.high_confidence_errors, ())
        self.assertEqual([item["sample_id"] for item in result.most_uncertain], [1, 4, 5])
        self.assertEqual(result.summary["correct_confident"], 3)
        self.assertTrue(all(item["uncertainty"] == result.most_uncertain[0]["uncertainty"] for item in result.most_uncertain))

    def test_shuffled_features_join_by_sample_id(self):
        features = pd.DataFrame({"age": [10, 20, 30]}, index=[1, 2, 3])
        store = _store([3, 1, 2], [0, 1, 1], [1, 1, 1], [0.9, 0.2, 0.4])
        shuffled = features.loc[[2, 3, 1]]
        result = explore_hard_examples(store, features=shuffled, top_n=3)
        by_id = {item["sample_id"]: item["features"]["age"] for item in result.most_uncertain}
        self.assertEqual(by_id, {1: 10, 2: 20, 3: 30})

    def test_json_uses_the_hard_example_directory(self):
        store = _store([1], [0], [1], [0.9])
        result = explore_hard_examples(
            store,
            dataset="breast_cancer",
            model="logistic_regression",
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = hard_examples_directory(Path(tmp) / "outputs" / "experiments" / "run")
            path = save_hard_examples(result, directory)
            self.assertEqual(
                path,
                Path(tmp)
                / "outputs"
                / "diagnostics"
                / "hard_examples"
                / "breast_cancer_logistic_regression_hard_examples.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("high_confidence_errors", payload)
        self.assertIn("most_uncertain", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["summary"]["wrong_confident"], 1)


def _store(sample_ids, y_true, y_pred, probability, features=None) -> PredictionStore:
    return PredictionStore.from_predictions(
        y_true,
        y_pred,
        probability,
        sample_ids=sample_ids,
        features=features,
    )
