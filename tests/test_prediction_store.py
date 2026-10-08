"""Tests for the standardized prediction store."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.diagnostics.prediction_store import (
    PREDICTION_COLUMNS,
    PredictionStore,
    prediction_directory,
    save_experiment_predictions,
)


class BinaryPredictionTests(unittest.TestCase):
    def test_positive_class_probability_and_confidence(self):
        labels = pd.Series([0, 1, 1, 0], index=[10, 4, 7, 2])
        predicted = [0, 1, 0, 1]
        positive = [0.2, 0.9, 0.4, 1.0]
        store = PredictionStore.from_predictions(labels, predicted, positive)

        frame = store.to_frame()
        self.assertEqual(list(frame.columns), list(PREDICTION_COLUMNS))
        self.assertEqual(frame["sample_id"].tolist(), [10, 4, 7, 2])
        self.assertEqual(frame["probability"].tolist(), positive)
        self.assertEqual(frame["confidence"].tolist(), [0.8, 0.9, 0.6, 1.0])
        self.assertEqual(frame["correct"].tolist(), [True, True, False, False])

    def test_binary_probability_matrix_uses_class_one(self):
        matrix = np.asarray(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )
        store = PredictionStore.from_predictions(
            [0, 1],
            [0, 1],
            matrix,
            classes=[1, 0],
            sample_ids=[5, 9],
        )
        frame = store.to_frame()
        self.assertEqual(frame["probability"].tolist(), [0.8, 0.1])
        self.assertEqual(frame["confidence"].tolist(), [0.8, 0.9])

    def test_feature_rows_join_on_sample_id(self):
        features = pd.DataFrame(
            {"mean radius": [11.0, 15.0]},
            index=[7, 3],
        )
        store = PredictionStore.from_predictions(
            pd.Series([1, 0], index=[3, 7]),
            [1, 0],
            [0.7, 0.25],
            features=features,
        )
        joined = store.to_frame(include_features=True)
        self.assertEqual(joined["sample_id"].tolist(), [3, 7])
        self.assertEqual(joined["mean radius"].tolist(), [15.0, 11.0])

    def test_model_result_and_experiment_keep_the_test_index(self):
        y_test = pd.Series([0, 1], index=[8, 1], name="target")
        features = pd.DataFrame({"age": [40, 22]}, index=y_test.index)
        model_result = SimpleNamespace(
            predictions=np.asarray([1, 1]),
            positive_class_probabilities=np.asarray([0.6, 0.7]),
        )
        from_result = PredictionStore.from_model_result(
            model_result,
            y_test,
            features=features,
        )
        self.assertEqual(from_result.frame["sample_id"].tolist(), [8, 1])
        self.assertEqual(from_result.frame["correct"].tolist(), [False, True])

        experiment = SimpleNamespace(
            models=("logistic_regression",),
            y_test=y_test,
            X_test=features,
            fitted={"logistic_regression": model_result},
        )
        from_experiment = PredictionStore.from_experiment(
            experiment,
            "logistic_regression",
        )
        self.assertEqual(from_experiment.frame["sample_id"].tolist(), [8, 1])
        restored = from_experiment.join_features(features)
        self.assertEqual(restored["age"].tolist(), [40, 22])

    def test_phase2_prediction_file_uses_test_index(self):
        payload = {
            "dataset": "breast_cancer",
            "test_index": [4, 9],
            "y_test": [1, 0],
            "models": {
                "logistic_regression": {
                    "predictions": [1, 0],
                    "positive_class_probabilities": [0.8, 0.3],
                }
            },
        }
        store = PredictionStore.from_phase2_predictions(payload, "logistic_regression")
        frame = store.to_frame()
        self.assertEqual(frame["sample_id"].tolist(), [4, 9])
        self.assertEqual(frame["confidence"].tolist(), [0.8, 0.7])
        self.assertEqual(frame["correct"].tolist(), [True, True])


class MulticlassPredictionTests(unittest.TestCase):
    def test_predicted_class_probability_and_max_confidence(self):
        matrix = np.asarray(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
                [0.2, 0.5, 0.3],
            ]
        )
        store = PredictionStore.from_predictions(
            [2, 0, 2],
            [2, 0, 2],
            matrix,
            classes=[0, 1, 2],
            sample_ids=["a", "b", "c"],
        )
        frame = store.to_frame()
        self.assertEqual(frame["probability"].tolist(), [0.7, 0.5, 0.3])
        self.assertEqual(frame["confidence"].tolist(), [0.7, 0.5, 0.5])
        self.assertEqual(frame["correct"].tolist(), [True, True, True])


class MalformedPredictionTests(unittest.TestCase):
    def test_length_mismatch_and_invalid_probability(self):
        with self.assertRaises(ValueError) as mismatch:
            PredictionStore.from_predictions([0, 1], [0], [0.2, 0.8])
        self.assertIn("different lengths", str(mismatch.exception))

        with self.assertRaises(ValueError) as invalid:
            PredictionStore.from_predictions([0, 1], [0, 1], [0.2, 1.2])
        self.assertIn("probability", str(invalid.exception))

        with self.assertRaises(ValueError) as duplicate:
            PredictionStore.from_predictions(
                [0, 1],
                [0, 1],
                [0.2, 0.8],
                sample_ids=[3, 3],
            )
        self.assertIn("sample_id", str(duplicate.exception))

        with self.assertRaises(ValueError) as missing_model:
            PredictionStore.from_experiment(
                SimpleNamespace(models=("logistic_regression",), fitted={}),
                "random_forest",
            )
        self.assertIn("random_forest", str(missing_model.exception))
        self.assertIn("logistic_regression", str(missing_model.exception))

    def test_saved_file_must_contain_required_columns_and_consistent_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            pd.DataFrame(
                {
                    "sample_id": [1],
                    "y_true": [0],
                    "y_pred": [1],
                    "probability": [0.4],
                    "confidence": [0.6],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError) as missing:
                PredictionStore.load_csv(path)
            self.assertIn("correct", str(missing.exception))

            pd.DataFrame(
                {
                    "sample_id": [1],
                    "y_true": [0],
                    "y_pred": [1],
                    "probability": [1.4],
                    "confidence": [0.6],
                    "correct": [False],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError) as invalid:
                PredictionStore.load_csv(path)
            self.assertIn("probability", str(invalid.exception))

            pd.DataFrame(
                {
                    "sample_id": [1],
                    "y_true": [0],
                    "y_pred": [1],
                    "probability": [0.4],
                    "confidence": [0.6],
                    "correct": [True],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError) as inconsistent:
                PredictionStore.load_csv(path)
            self.assertIn("correct", str(inconsistent.exception))


class PredictionExportTests(unittest.TestCase):
    def test_saved_csv_comes_from_stored_predictions(self):
        y_test = pd.Series([0, 1], index=[4, 8])
        features = pd.DataFrame({"age": [30, 40]}, index=y_test.index)
        model = SimpleNamespace(
            predictions=np.asarray([0, 1]),
            positive_class_probabilities=np.asarray([0.2, 0.8]),
        )
        result = SimpleNamespace(
            dataset="breast_cancer",
            models=("logistic_regression",),
            y_test=y_test,
            X_test=features,
            fitted={"logistic_regression": model},
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "outputs" / "experiments" / "run"
            written = save_experiment_predictions(result, model_dir)
            path = written["logistic_regression"]
            self.assertEqual(
                path,
                Path(tmp) / "outputs" / "predictions" / "breast_cancer_logistic_regression_predictions.csv",
            )
            self.assertEqual(
                prediction_directory(model_dir.parent),
                Path(tmp) / "outputs" / "predictions",
            )
            loaded = PredictionStore.load_csv(path)
        self.assertEqual(loaded.frame["sample_id"].tolist(), [4, 8])
        self.assertEqual(loaded.frame["probability"].tolist(), [0.2, 0.8])
        self.assertEqual(loaded.frame["confidence"].tolist(), [0.8, 0.8])
        joined = loaded.join_features(features)
        self.assertEqual(joined["age"].tolist(), [30, 40])


class PredictionFileTests(unittest.TestCase):
    def test_csv_round_trip_keeps_prediction_columns(self):
        store = PredictionStore.from_predictions(
            pd.Series([1, 0], index=[6, 2]),
            [0, 0],
            [0.45, 0.1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = store.save_csv(Path(tmp) / "nested" / "predictions.csv")
            loaded = PredictionStore.load_csv(path)
        frame = loaded.to_frame()
        self.assertEqual(list(frame.columns), list(PREDICTION_COLUMNS))
        self.assertEqual(frame["sample_id"].tolist(), [6, 2])
        self.assertEqual(frame["y_true"].tolist(), [1, 0])
        self.assertEqual(frame["y_pred"].tolist(), [0, 0])
        self.assertTrue(np.allclose(frame["probability"], [0.45, 0.1]))
        self.assertTrue(np.allclose(frame["confidence"], [0.55, 0.9]))
        self.assertEqual(frame["correct"].tolist(), [False, True])
        self.assertIsNone(loaded.features)
