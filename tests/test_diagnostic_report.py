"""Failure reports are rendered from stored diagnostic results."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.diagnostics.report import (
    build_robustness_report,
    render_failure_analysis,
    write_diagnostic_reports,
)
from src.experiments.config import ExperimentConfig
from src.diagnostics.diagnostic_engine import run_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class DiagnosticReportTests(unittest.TestCase):
    def test_report_generation_writes_markdown_and_json(self):
        payload = _complete_payload()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_diagnostic_reports(payload, directory=tmp)
            markdown = written.markdown_path.read_text(encoding="utf-8")
            robustness = json.loads(written.robustness_path.read_text(encoding="utf-8"))
        self.assertEqual(written.markdown_path.name, "failure_analysis.md")
        self.assertEqual(written.robustness_path.name, "robustness_report.json")
        self.assertIn("# Model Failure Analysis", markdown)
        recorded = next(row for row in robustness["comparison"] if row["f1_drop"] == 0.123456789)
        self.assertEqual(recorded["model"], "linear")
        self.assertNotIn("0.1235", json.dumps(recorded["f1_drop"]))

    def test_report_does_not_recompute_metrics(self):
        with patch(
            "src.evaluation.evaluate.score_predictions",
            side_effect=AssertionError("recomputed"),
        ):
            markdown = render_failure_analysis(_complete_payload())
        self.assertIn("## Overall Performance", markdown)
        self.assertIn("0.9100", markdown)

    def test_missing_diagnostic_components_are_stated(self):
        markdown = render_failure_analysis(_run_payload(_experiment_only()))
        for heading in (
            "## 1. Failure Slices",
            "## 2. High-Confidence Errors",
            "## 3. Most Uncertain Predictions",
            "## 4. Robustness",
            "## 5. Feature Importance",
            "## 6. Diagnostic Summary",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("This component was not included in the diagnostic result.", markdown)
        self.assertIn("There is insufficient evidence to list failure slices.", markdown)
        self.assertIn("There is insufficient evidence to describe important features.", markdown)
        self.assertNotIn("causes the model to fail", markdown)

    def test_unsupported_explanation_keeps_the_other_sections(self):
        run = _supported_run("sample_data", "linear")
        run["explanation"] = _component(
            "explanation",
            "not_supported",
            "This model does not expose feature_importances_.",
            {
                "status": "not_supported",
                "reason": "This model does not expose feature_importances_.",
                "features": [],
            },
        )
        run["diagnostic_summary"] = _component(
            "diagnostic_summary",
            "not_supported",
            "This model does not expose feature_importances_.",
            {"associations": [], "note": "Associations only."},
        )
        markdown = render_failure_analysis(_run_payload(run))
        self.assertIn("Feature importance is not supported.", markdown)
        self.assertIn("This model does not expose feature_importances_.", markdown)
        self.assertIn("There is insufficient evidence to describe important features.", markdown)
        self.assertIn("width", markdown)
        self.assertIn("## 1. Failure Slices", markdown)
        self.assertIn("Q4: (60, 90]", markdown)

    def test_empty_slices_state_insufficient_evidence(self):
        run = _experiment_only()
        run["slicing"] = _component(
            "slicing",
            "insufficient_data",
            "Error slicing produced no slices.",
            {"slices": [], "worst_slices": [], "overall_metrics": {}},
        )
        markdown = render_failure_analysis(_run_payload(run))
        self.assertIn("There is insufficient evidence to list failure slices.", markdown)
        self.assertIn(
            "There is insufficient evidence to identify the largest failure slice.",
            markdown,
        )
        self.assertNotIn("| feature | slice |", markdown)

    def test_empty_hard_examples_state_that_none_were_stored(self):
        run = _experiment_only()
        run["hard_examples"] = _component(
            "hard_examples",
            "supported",
            None,
            {
                "high_confidence_errors": [],
                "most_uncertain": [],
                "summary": {
                    "n_samples": 4,
                    "n_errors": 0,
                    "correct_confident": 4,
                    "correct_uncertain": 0,
                    "wrong_confident": 0,
                    "wrong_uncertain": 0,
                },
            },
        )
        markdown = render_failure_analysis(_run_payload(run))
        self.assertIn("No high-confidence errors were stored.", markdown)
        self.assertIn("No uncertain predictions were stored.", markdown)
        self.assertIn("no notable high-confidence errors", markdown)
        self.assertIn("0 correct uncertain predictions", markdown)

    def test_robustness_only_results_leave_other_sections_explicit(self):
        run = _experiment_only()
        run["robustness"] = _component(
            "robustness",
            "supported",
            None,
            _robustness(0.2, 0.05),
        )
        markdown = render_failure_analysis(_run_payload(run))
        robustness = build_robustness_report(_run_payload(run))
        self.assertIn("## 4. Robustness", markdown)
        self.assertIn("mean F1", markdown)
        self.assertIn("0.2000", markdown)
        self.assertIn("0.0500", markdown)
        self.assertIn("There is insufficient evidence to list failure slices.", markdown)
        self.assertIn("There is insufficient evidence to list high-confidence errors.", markdown)
        self.assertIn("There is insufficient evidence to list ranked features.", markdown)
        noisy = robustness["comparison"][1]
        self.assertEqual(noisy["noise_level"], 0.1)
        self.assertEqual(noisy["f1_mean"], 0.8 - 0.2)
        self.assertEqual(noisy["f1_std"], 0.05)
        self.assertEqual(noisy["f1_drop"], 0.2)
        self.assertEqual(robustness["runs"][0]["status"], "supported")

    def test_complete_report_covers_every_section_and_compares_models(self):
        markdown = render_failure_analysis(_complete_payload())
        robustness = build_robustness_report(_complete_payload())
        for heading in (
            "# Model Failure Analysis",
            "## Experiment",
            "## Overall Performance",
            "## 1. Failure Slices",
            "## 2. High-Confidence Errors",
            "## 3. Most Uncertain Predictions",
            "## 4. Robustness",
            "## 5. Feature Importance",
            "## 6. Diagnostic Summary",
            "### Comparison",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("sample_data / linear", markdown)
        self.assertIn("sample_data / tree", markdown)
        self.assertIn("random_state: 42", markdown)
        self.assertIn("0 = negative", markdown)
        self.assertIn("1 = positive", markdown)
        self.assertIn("Q4: (60, 90]", markdown)
        self.assertIn("sample count", markdown)
        self.assertIn("delta F1", markdown)
        self.assertIn("17", markdown)
        self.assertIn("0 (negative)", markdown)
        self.assertIn("1 (positive)", markdown)
        self.assertIn("group", markdown)
        self.assertIn("east", markdown)
        self.assertIn("probability", markdown)
        self.assertIn("uncertainty", markdown)
        self.assertIn("performance drop", markdown)
        self.assertIn("tree mean F1", markdown)
        self.assertIn("linear mean F1", markdown)
        self.assertIn("The largest failure slice is width", markdown)
        self.assertIn("The largest performance degradation", markdown)
        self.assertIn("notable high-confidence error", markdown)
        uncertain = render_failure_analysis(_uncertain_error_payload())
        self.assertNotIn("notable high-confidence error", uncertain)
        self.assertIn("no incorrect predictions at or above the confidence threshold", uncertain)
        self.assertIn("highest-confidence incorrect prediction", uncertain)
        self.assertIn("most uncertain stored prediction", markdown)
        self.assertIn("largest robustness degradation", markdown)
        self.assertIn("highest ranked important feature is width", markdown)
        self.assertIn("Feature importance is not supported.", markdown)
        self.assertIn(
            "width is among the model's most important features, "
            "while the slice Q4: (60, 90] exhibits substantial performance degradation.",
            markdown,
        )
        self.assertIn("do not show that a feature caused the model to fail", markdown)
        self.assertNotIn("causes the model to fail", markdown)
        self.assertNotIn("mean radius", markdown)
        self.assertEqual(
            [row["model"] for row in robustness["comparison"]],
            ["linear", "linear", "tree", "tree"],
        )
        self.assertEqual(robustness["comparison"][3]["f1_drop"], 0.25)

    def test_saved_engine_result_renders_without_changing_phase1(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ExperimentConfig(
                dataset="breast_cancer",
                models=("logistic_regression", "random_forest"),
                random_state=42,
                test_size=0.2,
                output_dir=root / "outputs" / "experiments",
            )
            result = run_diagnostics(config, noise_levels=(0.0,), n_seeds=1)
            written = write_diagnostic_reports(result, directory=root / "diagnostics")
            markdown = written.markdown_path.read_text(encoding="utf-8")
            robustness = json.loads(written.robustness_path.read_text(encoding="utf-8"))
        by_model = {run.experiment["model"]: run for run in result.runs}
        logistic = by_model["logistic_regression"]
        forest = by_model["random_forest"]
        feature = forest.explanation.result["features"][0]["feature"]
        stored_drop = forest.robustness.result["results"][0]["degradation"]["f1"]
        stored_mean = forest.robustness.result["results"][0]["mean"]["f1"]
        self.assertIn("Feature importance is not supported.", markdown)
        self.assertIn(feature, markdown)
        self.assertIn("### Comparison", markdown)
        self.assertIn("do not show that a feature caused the model to fail", markdown)
        self.assertNotIn("causes the model to fail", markdown)
        matched = [
            row
            for row in robustness["comparison"]
            if row["model"] == "random_forest" and row["noise_level"] == 0.0
        ]
        self.assertEqual(matched[0]["f1_mean"], stored_mean)
        self.assertEqual(matched[0]["f1_drop"], stored_drop)
        logistic_report = next(
            row for row in robustness["runs"] if row["model"] == "logistic_regression"
        )
        self.assertEqual(logistic.explanation.status, "not_supported")
        self.assertEqual(
            logistic_report["baseline_metrics"]["accuracy"],
            logistic.robustness.result["baseline_metrics"]["accuracy"],
        )
        self.assertEqual(_artifact_bytes(), before)


def _complete_payload() -> dict:
    linear = _supported_run("sample_data", "linear")
    linear["explanation"] = _component(
        "explanation",
        "not_supported",
        "This model does not expose feature_importances_.",
        {
            "status": "not_supported",
            "reason": "This model does not expose feature_importances_.",
            "features": [],
        },
    )
    linear["diagnostic_summary"] = _component(
        "diagnostic_summary",
        "not_supported",
        "This model does not expose feature_importances_.",
        {"associations": []},
    )
    linear["robustness"] = _component("robustness", "supported", None, _robustness(0.123456789, 0.01))
    tree = _supported_run("sample_data", "tree")
    tree["baseline_metrics"] = {
        "accuracy": 0.88,
        "precision": 0.86,
        "recall": 0.9,
        "f1": 0.88,
    }
    tree["robustness"] = _component("robustness", "supported", None, _robustness(0.25, 0.02))
    tree["explanation"] = _component(
        "explanation",
        "supported",
        None,
        {
            "status": "ok",
            "features": [
                {"feature": "width", "importance": 0.7, "rank": 1},
                {"feature": "group", "importance": 0.3, "rank": 2},
            ],
        },
    )
    tree["diagnostic_summary"] = _component(
        "diagnostic_summary",
        "supported",
        None,
        {
            "association_only": True,
            "associations": [
                {
                    "statement": (
                        "width is among the model's most important features, "
                        "while the slice Q4: (60, 90] exhibits substantial performance degradation."
                    )
                }
            ],
        },
    )
    return {
        "datasets": ["sample_data"],
        "models": ["linear", "tree"],
        "random_state": 42,
        "test_size": 0.2,
        "output_dir": "outputs/experiments",
        "runs": [linear, tree],
    }


def _supported_run(dataset: str, model: str) -> dict:
    run = _experiment_only(dataset, model)
    run["baseline_metrics"] = {
        "accuracy": 0.91,
        "precision": 0.9,
        "recall": 0.92,
        "f1": 0.91,
        "roc_auc": 0.95,
    }
    run["slicing"] = _component(
        "slicing",
        "supported",
        None,
        {
            "note": (
                "Slice metrics describe performance differences within subgroups of the "
                "test set. They do not show that a feature caused the model to fail."
            ),
            "overall_metrics": {"accuracy": 0.91, "f1": 0.91},
            "worst_slices": [
                {
                    "feature": "width",
                    "slice_definition": "Q4: (60, 90]",
                    "sample_count": 24,
                    "metrics": {"accuracy": 0.7, "f1": 0.62},
                    "delta_accuracy": -0.21,
                    "delta_f1": -0.29,
                }
            ],
        },
    )
    run["hard_examples"] = _component(
        "hard_examples",
        "supported",
        None,
        {
            "summary": {
                "n_samples": 20,
                "n_errors": 2,
                "correct_confident": 16,
                "correct_uncertain": 2,
                "wrong_confident": 1,
                "wrong_uncertain": 1,
                "confidence_threshold": 0.75,
            },
            "high_confidence_errors": [
                {
                    "sample_id": 17,
                    "y_true": 0,
                    "y_pred": 1,
                    "probability": 0.93,
                    "confidence": 0.93,
                    "uncertainty": 0.14,
                    "features": {"width": 61.5, "group": "east"},
                }
            ],
            "most_uncertain": [
                {
                    "sample_id": 8,
                    "y_true": 1,
                    "y_pred": 0,
                    "probability": 0.51,
                    "confidence": 0.51,
                    "uncertainty": 0.98,
                    "features": {"width": 40.0, "group": "west"},
                }
            ],
        },
    )
    run["robustness"] = _component("robustness", "supported", None, _robustness(0.123456789, 0.01))
    run["explanation"] = _component(
        "explanation",
        "supported",
        None,
        {"features": [{"feature": "width", "importance": 0.7, "rank": 1}]},
    )
    run["diagnostic_summary"] = _component(
        "diagnostic_summary",
        "supported",
        None,
        {"associations": []},
    )
    return run


def _experiment_only(dataset: str = "sample_data", model: str = "linear") -> dict:
    return {
        "experiment": {
            "dataset": dataset,
            "model": model,
            "experiment_id": f"{dataset}__{model}__rs42__ts0p2",
            "random_state": 42,
            "test_size": 0.2,
            "class_labels": {"0": "negative", "1": "positive"},
            "numeric_features": ["width"],
            "categorical_features": ["group"],
            "n_test_samples": 20,
            "reused_artifacts": True,
        },
        "baseline_metrics": None,
    }


def _run_payload(run: dict) -> dict:
    return {
        "datasets": [run["experiment"]["dataset"]],
        "models": [run["experiment"]["model"]],
        "random_state": 42,
        "test_size": 0.2,
        "runs": [run],
    }


def _uncertain_error_payload() -> dict:
    payload = _complete_payload()
    for run in payload["runs"]:
        stored = run["hard_examples"]["result"]
        stored["summary"]["wrong_confident"] = 0
        stored["summary"]["n_errors"] = 1
        stored["high_confidence_errors"] = [
            {
                "sample_id": 4,
                "y_true": 0,
                "y_pred": 1,
                "probability": 0.28,
                "confidence": 0.72,
                "uncertainty": 0.56,
                "category": "wrong_uncertain",
                "features": {"width": 12.0},
            }
        ]
    return payload


def _component(name: str, status: str, detail, result) -> dict:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "output_path": None,
        "result": result,
    }


def _robustness(drop: float, spread: float) -> dict:
    return {
        "baseline_metrics": {"f1": 0.8},
        "noise_configuration": {"noise_levels": [0.0, 0.1]},
        "results": [
            {
                "noise_level": 0.0,
                "mean": {"f1": 0.8},
                "std": {"f1": 0.0},
                "degradation": {"f1": 0.0},
            },
            {
                "noise_level": 0.1,
                "mean": {"f1": 0.8 - drop},
                "std": {"f1": spread},
                "degradation": {"f1": drop},
            },
        ],
    }


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS}
