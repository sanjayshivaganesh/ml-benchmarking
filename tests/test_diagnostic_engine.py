"""The diagnostic engine runs every Phase 3 check from one configuration."""

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.data_loader import SUPPORTED_DATASETS
from src.diagnostics.diagnostic_engine import (
    STATUS_FAILED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_NOT_SUPPORTED,
    STATUS_SUPPORTED,
    DiagnosticComponent,
    _component,
    _validate_model,
    run_diagnostics,
)
from src.evaluation.evaluate import LABEL_METRIC_NAMES
from src.experiments.config import ExperimentConfig
from src.run_diagnostics import main as diagnostics_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class DiagnosticEngineTests(unittest.TestCase):
    def test_component_failures_are_explicit(self):
        blocked = DiagnosticComponent("prediction_validation", STATUS_FAILED, "labels differ")
        skipped = _component("slicing", blocked, _raise_value)
        self.assertEqual(skipped.name, "slicing")
        self.assertEqual(skipped.status, STATUS_FAILED)
        self.assertEqual(skipped.detail, "labels differ")

        failed = _component("slicing", None, _raise_value)
        self.assertEqual(failed.status, STATUS_FAILED)
        self.assertEqual(failed.detail, "bad slice")

        with self.assertRaises(RuntimeError):
            _component("slicing", None, _raise_runtime)

    def test_prediction_validation_distinguishes_bad_rows(self):
        prepared = _prepared()
        supported = _validate_model(prepared, None, "logistic_regression")
        self.assertEqual(supported.status, STATUS_SUPPORTED)

        missing = _validate_model(prepared, None, "random_forest")
        self.assertEqual(missing.status, STATUS_FAILED)
        self.assertIn("random_forest", missing.detail)

        empty = _prepared(predictions=[])
        insufficient = _validate_model(empty, None, "logistic_regression")
        self.assertEqual(insufficient.status, STATUS_INSUFFICIENT_DATA)

        mismatched = _prepared()
        mismatched["labels"] = pd.Series([1, 0], index=[0, 1])
        failed = _validate_model(mismatched, None, "logistic_regression")
        self.assertEqual(failed.status, STATUS_FAILED)

    def test_unknown_dataset_is_rejected_before_training(self):
        config = _config("breast_cancer", ("logistic_regression",), Path("unused"))
        with patch(
            "src.diagnostics.diagnostic_engine.run_phase2",
            side_effect=AssertionError("trained"),
        ):
            with self.assertRaises(ValueError) as caught:
                run_diagnostics(config, datasets=("breast_cancer", "titanic"))
        self.assertIn("Available datasets", str(caught.exception))
        self.assertIn("breast_cancer", str(caught.exception))
        self.assertIn("adult", str(caught.exception))

    def test_saved_experiment_diagnoses_every_selected_model(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(
                "breast_cancer",
                ("logistic_regression", "random_forest"),
                Path(tmp) / "outputs" / "experiments",
            )
            first = run_diagnostics(config, noise_levels=(0.0,), n_seeds=1)
            self.assertEqual(
                [run.experiment["model"] for run in first.runs],
                ["logistic_regression", "random_forest"],
            )
            self.assertTrue(all(run.experiment["reused_artifacts"] is False for run in first.runs))
            by_model = {run.experiment["model"]: run for run in first.runs}
            logistic = by_model["logistic_regression"]
            forest = by_model["random_forest"]
            for run in (logistic, forest):
                self.assertEqual(run.prediction_validation.status, STATUS_SUPPORTED)
                self.assertEqual(run.slicing.status, STATUS_SUPPORTED)
                self.assertEqual(run.hard_examples.status, STATUS_SUPPORTED)
                self.assertEqual(run.robustness.status, STATUS_SUPPORTED)
                self.assertEqual(run.slicing.result["overall_metrics"]["accuracy"], run.baseline_metrics["accuracy"])
                self.assertIn("mean radius", run.experiment["numeric_features"])
            self.assertEqual(logistic.explanation.status, STATUS_NOT_SUPPORTED)
            self.assertEqual(logistic.diagnostic_summary.status, STATUS_NOT_SUPPORTED)
            self.assertEqual(forest.explanation.status, STATUS_SUPPORTED)
            self.assertEqual(forest.diagnostic_summary.status, STATUS_SUPPORTED)
            self.assertTrue(forest.diagnostic_summary.result["association_only"])
            self.assertIn("mean radius", [row["feature"] for row in forest.explanation.result["features"]])
            for name in LABEL_METRIC_NAMES:
                self.assertEqual(
                    forest.slicing.result["overall_metrics"][name],
                    forest.baseline_metrics[name],
                )
            with patch(
                "src.experiments.experiment.train_one",
                side_effect=AssertionError("retrained"),
            ):
                second = run_diagnostics(config, noise_levels=(0.0,), n_seeds=1)
            self.assertTrue(all(run.experiment["reused_artifacts"] is True for run in second.runs))
            self.assertEqual(
                [run.explanation.status for run in second.runs],
                [STATUS_NOT_SUPPORTED, STATUS_SUPPORTED],
            )
        self.assertEqual(_artifact_bytes(), before)

    def test_multiple_datasets_use_the_same_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(
                "breast_cancer",
                ("logistic_regression",),
                Path(tmp) / "outputs" / "experiments",
            )
            result = run_diagnostics(
                config,
                datasets=("breast_cancer", "adult"),
                noise_levels=(0.0,),
                n_seeds=1,
            )
        self.assertEqual(result.datasets, ("breast_cancer", "adult"))
        self.assertEqual(result.models, ("logistic_regression",))
        self.assertEqual(
            [run.experiment["dataset"] for run in result.runs],
            ["breast_cancer", "adult"],
        )
        adult = result.runs[1]
        self.assertIn("age", adult.experiment["numeric_features"])
        self.assertIn("workclass", adult.experiment["categorical_features"])
        self.assertEqual(adult.explanation.status, STATUS_NOT_SUPPORTED)
        self.assertEqual(adult.slicing.status, STATUS_SUPPORTED)
        self.assertEqual(adult.robustness.status, STATUS_SUPPORTED)
        self.assertNotIn("titanic", SUPPORTED_DATASETS)

    def test_cli_rejects_an_unknown_dataset_without_a_traceback(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = diagnostics_main(["--dataset", "titanic", "--models", "logistic_regression"])
        self.assertEqual(code, 1)
        self.assertIn("Available datasets", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")


def _raise_value():
    raise ValueError("bad slice")


def _raise_runtime():
    raise RuntimeError("bug")


def _config(dataset: str, models: tuple[str, ...], output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset,
        models=models,
        random_state=42,
        test_size=0.2,
        output_dir=output_dir,
    )


def _prepared(predictions=None):
    if predictions is None:
        predictions = [0, 1]
    return {
        "predictions": {
            "test_index": [0, 1],
            "y_test": [0, 1],
            "models": {
                "logistic_regression": {
                    "predictions": predictions,
                    "positive_class_probabilities": [0.2, 0.8][: len(predictions)],
                }
            },
        },
        "labels": pd.Series([0, 1][: len(predictions)], index=[0, 1][: len(predictions)]),
    }


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS}
