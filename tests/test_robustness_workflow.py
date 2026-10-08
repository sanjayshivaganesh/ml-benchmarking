"""Robustness testing loads a saved model and repeats exactly."""

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.data.data_loader import SUPPORTED_DATASETS
from src.diagnostics.robustness import compare_robustness_files
from src.diagnostics.workflow import run_diagnostic_robustness
from src.evaluation.evaluate import LABEL_METRIC_NAMES
from src.experiments.config import ExperimentConfig
from src.run_robustness import main as robustness_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class RobustnessWorkflowTests(unittest.TestCase):
    def test_saved_models_repeat_without_retraining(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                dataset="breast_cancer",
                models=("logistic_regression", "random_forest"),
                random_state=42,
                test_size=0.2,
                output_dir=Path(tmp) / "outputs" / "experiments",
            )
            first = run_diagnostic_robustness(config)
            self.assertFalse(first.reused_model)
            blobs = {name: path.read_bytes() for name, path in first.output_paths.items()}
            metrics_bytes = (first.experiment_dir / "metrics.json").read_bytes()
            for model_name, path in first.output_paths.items():
                self.assertEqual(path.name, f"breast_cancer_{model_name}_robustness.json")
                self.assertEqual(path.parent.name, "robustness")
                self.assertTrue((first.experiment_dir / f"breast_cancer__{model_name}.joblib").is_file())
                _assert_matches_experiment(self, path, first.experiment_dir, model_name)
            _assert_comparison(self, first.comparison, set(config.models))
            with patch("src.experiments.experiment.train_one", side_effect=AssertionError("retrained")):
                second = run_diagnostic_robustness(config)
            self.assertTrue(second.reused_model)
            self.assertEqual(second.output_paths, first.output_paths)
            for model_name, path in second.output_paths.items():
                self.assertEqual(path.read_bytes(), blobs[model_name])
            self.assertEqual((first.experiment_dir / "metrics.json").read_bytes(), metrics_bytes)
            reloaded = compare_robustness_files(tuple(second.output_paths.values()))
            pd_equal(self, reloaded, second.comparison)
        self.assertEqual(_artifact_bytes(), before)

    def test_adult_categorical_features_are_not_perturbed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                dataset="adult",
                models=("logistic_regression",),
                random_state=42,
                test_size=0.2,
                output_dir=Path(tmp) / "outputs" / "experiments",
            )
            first = run_diagnostic_robustness(config, noise_levels=(0.0, 0.1), n_seeds=1)
            path = first.output_paths["logistic_regression"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            perturbed = payload["noise_configuration"]["perturbed_features"]
            categorical = payload["noise_configuration"]["categorical_features"]
            self.assertIn("age", perturbed)
            self.assertIn("workclass", categorical)
            self.assertNotIn("workclass", perturbed)
            original = path.read_bytes()
            with patch("src.experiments.experiment.train_one", side_effect=AssertionError("retrained")):
                second = run_diagnostic_robustness(config, noise_levels=(0.0, 0.1), n_seeds=1)
            self.assertTrue(second.reused_model)
            self.assertEqual(path.read_bytes(), original)

    def test_unknown_dataset_is_rejected_without_a_traceback(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = robustness_main(["--dataset", "titanic", "--models", "logistic_regression"])
        self.assertEqual(code, 1)
        self.assertIn("Available datasets", stderr.getvalue())
        self.assertIn("breast_cancer", stderr.getvalue())
        self.assertIn("adult", stderr.getvalue())
        self.assertNotIn("titanic", SUPPORTED_DATASETS)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")


def _assert_matches_experiment(test: unittest.TestCase, path: Path, experiment_dir: Path, model_name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    test.assertEqual(payload["dataset"], "breast_cancer")
    test.assertEqual(payload["model"], model_name)
    test.assertIn("mean radius", payload["noise_configuration"]["perturbed_features"])
    test.assertEqual(payload["noise_configuration"]["seeds"], [42, 43, 44, 45, 46])
    zero = next(level for level in payload["results"] if level["noise_level"] == 0.0)
    for name in LABEL_METRIC_NAMES:
        test.assertEqual(payload["baseline_metrics"][name], metrics[model_name][name])
        test.assertEqual(zero["mean"][name], metrics[model_name][name])
        test.assertEqual(zero["degradation"][name], 0.0)


def _assert_comparison(test: unittest.TestCase, comparison, models: set[str]) -> None:
    test.assertEqual(set(comparison["model"]), models)
    test.assertIn("accuracy_mean", comparison.columns)
    test.assertIn("accuracy_drop", comparison.columns)
    test.assertEqual(len(comparison), len(models) * 5)


def pd_equal(test: unittest.TestCase, left, right) -> None:
    test.assertTrue(left.reset_index(drop=True).equals(right.reset_index(drop=True)))


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS}
