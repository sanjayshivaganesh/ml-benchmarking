"""Prediction CSVs written by a completed experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.diagnostics.prediction_store import PREDICTION_COLUMNS, PredictionStore
from src.evaluation.evaluate import METRIC_NAMES
from src.experiments.config import ExperimentConfig
from src.run_phase2 import run_phase2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class ExperimentPredictionTests(unittest.TestCase):
    def test_selected_models_write_prediction_csv_without_changing_metrics(self):
        before = {
            path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                dataset="breast_cancer",
                models=("logistic_regression", "random_forest"),
                random_state=42,
                test_size=0.2,
                output_dir=Path(tmp) / "outputs" / "experiments",
            )
            outcome = run_phase2(config)
            result = outcome.result
            prediction_dir = Path(tmp) / "outputs" / "predictions"
            written = {
                "logistic_regression": prediction_dir / "breast_cancer_logistic_regression_predictions.csv",
                "random_forest": prediction_dir / "breast_cancer_random_forest_predictions.csv",
            }
            self.assertFalse(
                (prediction_dir / "breast_cancer_gradient_boosting_predictions.csv").exists()
            )
            self.assertEqual(result.models, ("logistic_regression", "random_forest"))
            for model_name, path in written.items():
                self.assertTrue(path.is_file(), path)
                frame = PredictionStore.load_csv(path).to_frame()
                self.assertEqual(list(frame.columns), list(PREDICTION_COLUMNS))
                self.assertEqual(len(frame), len(result.y_test))
                self.assertEqual(frame["sample_id"].nunique(), len(frame))
                self.assertEqual(frame["sample_id"].tolist(), list(result.y_test.index))
                stored = result.fitted[model_name]
                self.assertTrue(
                    np.array_equal(frame["y_pred"].to_numpy(), stored.predictions)
                )
                self.assertTrue(
                    np.allclose(
                        frame["probability"].to_numpy(),
                        stored.positive_class_probabilities,
                    )
                )
                self.assertEqual(tuple(result.metrics[model_name]), METRIC_NAMES)
                self.assertTrue(
                    np.isclose(
                        frame["correct"].mean(),
                        result.metrics[model_name]["accuracy"],
                    )
                )
                joined = PredictionStore.load_csv(path).join_features(result.X_test)
                feature = result.feature_names[0]
                recovered = result.X_test.loc[frame["sample_id"].tolist(), feature]
                self.assertTrue(np.array_equal(joined[feature].to_numpy(), recovered.to_numpy()))
        self.assertEqual(
            {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS},
            before,
        )
