"""Diagnostics reuse saved experiment predictions."""

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import math

import numpy as np
import pandas as pd

from src.data.data_loader import SUPPORTED_DATASETS, load_dataset
from src.diagnostics.workflow import run_diagnostic_hard_examples, run_diagnostic_slicing
from src.experiments.config import ExperimentConfig
from src.run_diagnostics import main as diagnostics_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class DiagnosticReuseTests(unittest.TestCase):
    def test_existing_datasets_slice_without_retraining(self):
        self.assertEqual(SUPPORTED_DATASETS, ("breast_cancer", "adult"))
        before = _artifact_bytes()
        for dataset, expected_numeric, expected_categorical in (
            ("breast_cancer", "mean radius", None),
            ("adult", "age", "workclass"),
        ):
            with self.subTest(dataset=dataset):
                _assert_reuse(self, dataset, expected_numeric, expected_categorical)
        self.assertEqual(_artifact_bytes(), before)

    def test_unknown_dataset_is_rejected_without_a_traceback(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = diagnostics_main(["--dataset", "titanic", "--models", "logistic_regression"])
        self.assertEqual(code, 1)
        self.assertIn("Available datasets", stderr.getvalue())
        self.assertIn("breast_cancer", stderr.getvalue())
        self.assertIn("adult", stderr.getvalue())
        self.assertNotIn("titanic", SUPPORTED_DATASETS)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")


def _assert_reuse(test: unittest.TestCase, dataset: str, numeric_name: str, categorical_name: str | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = ExperimentConfig(
            dataset=dataset,
            models=("logistic_regression",),
            random_state=42,
            test_size=0.2,
            output_dir=Path(tmp) / "outputs" / "experiments",
        )
        first = run_diagnostic_slicing(config)
        test.assertFalse(first.reused_predictions)
        path = first.output_paths["logistic_regression"]
        test.assertEqual(path.name, f"{dataset}_logistic_regression_slices.json")
        path.unlink()
        with patch("src.experiments.experiment.train_one", side_effect=AssertionError("retrained")):
            second = run_diagnostic_slicing(config)
        test.assertTrue(second.reused_predictions)
        test.assertTrue(path.is_file())
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = json.loads((first.experiment_dir / "metrics.json").read_text(encoding="utf-8"))
        test.assertEqual(payload["dataset"], dataset)
        test.assertEqual(payload["model"], "logistic_regression")
        test.assertTrue(payload["association_only"])
        test.assertAlmostEqual(
            payload["overall_metrics"]["accuracy"],
            metrics["logistic_regression"]["accuracy"],
        )
        test.assertAlmostEqual(
            payload["overall_metrics"]["f1"],
            metrics["logistic_regression"]["f1"],
        )
        features = {row["feature"] for row in payload["slices"]}
        test.assertIn(numeric_name, features)
        numeric_rows = [row for row in payload["slices"] if row["feature"] == numeric_name]
        test.assertTrue(all(row["feature_type"] == "numeric" for row in numeric_rows))
        if categorical_name is not None:
            test.assertIn(categorical_name, features)
            categorical_rows = [
                row for row in payload["slices"] if row["feature"] == categorical_name
            ]
            test.assertTrue(all(row["feature_type"] == "categorical" for row in categorical_rows))
        _assert_hard_examples(test, config, first.experiment_dir, dataset)


def _assert_hard_examples(
    test: unittest.TestCase,
    config: ExperimentConfig,
    experiment_dir: Path,
    dataset: str,
) -> None:
    path = (
        experiment_dir.parent.parent
        / "diagnostics"
        / "hard_examples"
        / f"{dataset}_logistic_regression_hard_examples.json"
    )
    test.assertTrue(path.is_file())
    path.unlink()
    with patch("src.experiments.experiment.train_one", side_effect=AssertionError("retrained")):
        reused = run_diagnostic_hard_examples(config)
    test.assertTrue(reused.reused_predictions)
    test.assertEqual(reused.output_paths["logistic_regression"], path)
    test.assertTrue(path.is_file())
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = json.loads((experiment_dir / "predictions.json").read_text(encoding="utf-8"))
    split = load_dataset(dataset, test_size=config.test_size, random_state=config.random_state)
    saved = predictions["models"]["logistic_regression"]
    stored = {
        sample_id: {
            "y_true": y_true,
            "y_pred": y_pred,
            "probability": probability,
        }
        for sample_id, y_true, y_pred, probability in zip(
            predictions["test_index"],
            predictions["y_test"],
            saved["predictions"],
            saved["positive_class_probabilities"],
        )
    }
    test.assertEqual(payload["summary"]["n_samples"], len(stored))
    test.assertGreater(payload["summary"]["n_errors"], 0)
    errors = payload["high_confidence_errors"]
    test.assertEqual(
        [row["sample_id"] for row in errors],
        _ranked_ids(stored, kind="confidence"),
    )
    test.assertEqual(
        [row["confidence"] for row in errors],
        sorted((row["confidence"] for row in errors), reverse=True),
    )
    uncertain = payload["most_uncertain"]
    test.assertEqual(
        [row["sample_id"] for row in uncertain],
        _ranked_ids(stored, kind="uncertainty"),
    )
    for row in errors + uncertain:
        sample_id = row["sample_id"]
        test.assertIn(sample_id, split.X_test.index)
        test.assertEqual(row["y_true"], stored[sample_id]["y_true"])
        test.assertEqual(row["y_pred"], stored[sample_id]["y_pred"])
        test.assertEqual(row["correct"], row["y_true"] == row["y_pred"])
        original = split.X_test.loc[sample_id]
        test.assertEqual(list(row["features"]), list(split.X_test.columns))
        for column in split.X_test.columns:
            _assert_feature_value(test, row["features"][column], original[column])


def _ranked_ids(stored: dict, *, kind: str) -> list:
    ranked = []
    for sample_id, row in stored.items():
        probability = float(row["probability"])
        confidence = max(probability, 1.0 - probability)
        uncertainty = 1.0 - 2.0 * abs(probability - 0.5)
        ranked.append((sample_id, confidence, uncertainty, row["y_true"] == row["y_pred"]))
    if kind == "confidence":
        ranked = [item for item in ranked if not item[3]]
        ranked.sort(key=lambda item: (-item[1], item[0]))
    else:
        ranked.sort(key=lambda item: (-item[2], item[0]))
    return [item[0] for item in ranked[:20]]


def _assert_feature_value(test: unittest.TestCase, actual, expected) -> None:
    if _is_missing(expected):
        test.assertIsNone(actual)
        return
    if isinstance(expected, (float, np.floating)):
        test.assertTrue(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12))
        return
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, (bool, np.bool_)):
        test.assertEqual(actual, int(expected))
        return
    test.assertEqual(actual, expected)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS}
