"""Tests for a single dataset trained on selected baseline models."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from src.evaluation.evaluate import METRIC_NAMES
from src.experiments.experiment import DEFAULT_EXPERIMENT_DIR, run_experiment
from src.models.models import get_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS
    }


class RunExperimentTests(unittest.TestCase):
    def test_one_dataset_and_one_model(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                "breast_cancer",
                "logistic_regression",
                random_state=42,
                model_dir=tmp,
            )
            saved = list(Path(tmp).glob("*.joblib"))

        self.assertEqual(result.dataset, "breast_cancer")
        self.assertEqual(result.models, ("logistic_regression",))
        self.assertEqual(result.random_state, 42)
        self.assertEqual(result.test_size, 0.2)
        self.assertEqual(result.class_labels[0], "malignant")
        self.assertEqual(result.class_labels[1], "benign")
        self.assertEqual(set(result.metrics), {"logistic_regression"})
        self.assertEqual(tuple(result.metrics["logistic_regression"]), METRIC_NAMES)
        report = result.error_analysis["logistic_regression"]
        self.assertEqual(len(report["confusion_matrix"]), 2)
        self.assertIn("false_positives", report)
        self.assertIn("false_negatives", report)
        self.assertIn("misclassified_samples", report)
        self.assertIn("feature_slices", report)
        self.assertEqual(
            report["false_positives"] + report["false_negatives"],
            len(report["misclassified_samples"]),
        )
        fitted = result.fitted["logistic_regression"]
        self.assertEqual(
            fitted.trained.path.name,
            "breast_cancer__logistic_regression.joblib",
        )
        self.assertEqual(
            list(fitted.pipeline.named_steps),
            ["preprocessing", "classifier"],
        )
        check_is_fitted(fitted.pipeline)
        self.assertEqual(fitted.predictions.shape[0], len(result.y_test))
        self.assertEqual(fitted.positive_class_probabilities.shape[0], len(result.y_test))
        self.assertAlmostEqual(
            result.metrics["logistic_regression"]["roc_auc"],
            roc_auc_score(result.y_test, fitted.positive_class_probabilities),
        )
        self.assertTrue(result.metadata["stratified"])
        self.assertEqual(result.metadata["hyperparameters"]["logistic_regression"]["C"], 1.0)
        self.assertGreaterEqual(
            result.metadata["hyperparameters"]["logistic_regression"]["max_iter"],
            5000,
        )
        self.assertEqual(result.feature_names, list(result.X_test.columns))
        self.assertEqual(len(saved), 1)
        self.assertEqual(_artifact_bytes(), before)
        json.dumps(result.metrics)

    def test_one_dataset_and_two_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                "breast_cancer",
                ["logistic_regression", "random_forest"],
                random_state=42,
                model_dir=tmp,
            )
            produced = sorted(path.name for path in Path(tmp).glob("*.joblib"))

        self.assertEqual(result.models, ("logistic_regression", "random_forest"))
        self.assertEqual(
            list(result.metrics),
            ["logistic_regression", "random_forest"],
        )
        self.assertEqual(list(result.error_analysis), list(result.models))
        self.assertEqual(list(result.fitted), list(result.models))
        self.assertEqual(
            produced,
            [
                "breast_cancer__logistic_regression.joblib",
                "breast_cancer__random_forest.joblib",
            ],
        )

    def test_selected_model_filtering_trains_only_those_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                "breast_cancer",
                ("gradient_boosting", "logistic_regression"),
                random_state=42,
                model_dir=tmp,
            )
            produced = sorted(path.name for path in Path(tmp).glob("*.joblib"))

        self.assertEqual(result.models, ("gradient_boosting", "logistic_regression"))
        self.assertNotIn("random_forest", result.metrics)
        self.assertNotIn("random_forest", result.error_analysis)
        self.assertNotIn("random_forest", result.fitted)
        self.assertEqual(
            produced,
            [
                "breast_cancer__gradient_boosting.joblib",
                "breast_cancer__logistic_regression.joblib",
            ],
        )
        baselines = get_models(random_state=42)
        for model_name in result.models:
            self.assertEqual(
                result.fitted[model_name].pipeline.named_steps["classifier"].get_params(),
                baselines[model_name].get_params(),
            )

    def test_invalid_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment("not_a_dataset", "logistic_regression")

    def test_invalid_model_is_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment("breast_cancer", "support_vector_machine")

    def test_same_random_seed_reproduces_metrics(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_experiment(
                "breast_cancer",
                "random_forest",
                random_state=42,
                model_dir=first_dir,
            )
            second = run_experiment(
                "breast_cancer",
                "random_forest",
                random_state=42,
                model_dir=second_dir,
            )
        self.assertEqual(first.metrics, second.metrics)
        np.testing.assert_array_equal(
            first.fitted["random_forest"].predictions,
            second.fitted["random_forest"].predictions,
        )

    def test_default_directory_is_separate_from_phase1_outputs(self):
        self.assertEqual(DEFAULT_EXPERIMENT_DIR.parts[-2:], ("outputs", "experiments"))
        self.assertEqual(DEFAULT_EXPERIMENT_DIR, PROJECT_ROOT / "outputs" / "experiments")

    def test_batch_phase1_functions_are_not_used(self):
        import src.experiments.experiment as experiment_module

        self.assertNotIn("evaluate_all", experiment_module.__dict__)
        self.assertNotIn("analyze_all", experiment_module.__dict__)
        self.assertNotIn("run_phase1", experiment_module.__dict__)
