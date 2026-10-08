"""Model comparison uses stored diagnostic measurements only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.diagnostics.model_comparison import (
    HIGHER_IS_BETTER,
    HIGHER_IS_MORE_IMPORTANT_WITHIN_MODEL,
    LOWER_IS_BETTER,
    build_model_comparison,
    save_model_comparison,
)
from src.diagnostics.report import write_diagnostic_reports


class ModelComparisonTests(unittest.TestCase):
    def test_complete_comparison_ranks_stored_values(self):
        comparison = build_model_comparison(_payload())["comparisons"][0]
        metrics = {item["name"]: item for item in comparison["metrics"]}
        self.assertEqual(metrics["baseline_f1"]["direction"], HIGHER_IS_BETTER)
        self.assertEqual(metrics["worst_slice_delta_f1"]["direction"], HIGHER_IS_BETTER)
        self.assertEqual(metrics["high_confidence_errors"]["direction"], LOWER_IS_BETTER)
        self.assertEqual(metrics["mean_uncertainty"]["direction"], LOWER_IS_BETTER)
        self.assertEqual(metrics["robustness_f1_drop"]["direction"], LOWER_IS_BETTER)
        self.assertEqual(
            metrics["feature_importance"]["direction"],
            HIGHER_IS_MORE_IMPORTANT_WITHIN_MODEL,
        )
        self.assertFalse(metrics["feature_importance"]["comparable_across_models"])
        self.assertEqual(
            [row["value"] for row in metrics["baseline_f1"]["models"]],
            [0.91, 0.8],
        )
        self.assertEqual(comparison["findings"]["strongest_baseline_model"]["models"], ["linear"])
        self.assertEqual(
            comparison["findings"]["greatest_failure_slice_degradation"]["models"],
            ["tree"],
        )
        self.assertEqual(comparison["findings"]["greatest_failure_slice_degradation"]["value"], -0.4)
        self.assertEqual(comparison["findings"]["most_high_confidence_errors"]["models"], ["tree"])
        self.assertEqual(comparison["findings"]["most_high_confidence_errors"]["value"], 4)
        self.assertEqual(comparison["findings"]["most_robust_model"]["models"], ["linear"])
        self.assertEqual(comparison["findings"]["most_robust_model"]["noise_level"], 0.3)
        self.assertEqual(comparison["findings"]["most_robust_model"]["value"], 0.05)
        tree_importance = metrics["feature_importance"]["models"][1]
        self.assertEqual(tree_importance["status"], "supported")
        self.assertEqual(tree_importance["features"][0]["feature"], "width")
        linear_importance = metrics["feature_importance"]["models"][0]
        self.assertEqual(linear_importance["status"], "not_supported")
        self.assertEqual(linear_importance["features"], [])

    def test_tied_f1_deltas_keep_the_slicing_rank(self):
        payload = _payload()
        slices = payload["runs"][0]["slicing"]["result"]["worst_slices"]
        slices[:] = [
            {
                "feature": "education",
                "slice_definition": "1st-4th",
                "sample_count": 43,
                "delta_f1": -0.4,
                "delta_accuracy": 0.08,
            },
            {
                "feature": "native-country",
                "slice_definition": "Jamaica",
                "sample_count": 27,
                "delta_f1": -0.4,
                "delta_accuracy": -0.09,
            },
        ]
        comparison = build_model_comparison(payload)["comparisons"][0]
        metrics = {item["name"]: item for item in comparison["metrics"]}
        chosen = metrics["worst_slice_delta_f1"]["models"][0]
        self.assertEqual(chosen["feature"], "native-country")
        self.assertEqual(chosen["slice"], "Jamaica")
        self.assertEqual(chosen["value"], -0.4)

    def test_ties_and_the_highest_shared_noise_level_are_explicit(self):
        payload = _payload()
        payload["runs"][1]["baseline_metrics"]["f1"] = 0.91
        payload["runs"][0]["robustness"]["result"]["results"][1]["degradation"]["f1"] = 0.5
        comparison = build_model_comparison(payload)["comparisons"][0]
        baseline = comparison["findings"]["strongest_baseline_model"]
        self.assertTrue(baseline["tie"])
        self.assertEqual(baseline["models"], ["linear", "tree"])
        self.assertEqual(baseline["value"], 0.91)
        robust = comparison["findings"]["most_robust_model"]
        self.assertEqual(robust["models"], ["tree"])
        self.assertEqual(robust["noise_level"], 0.3)
        self.assertNotIn("causes", robust["selection"])

    def test_missing_measurements_are_not_invented(self):
        payload = _payload()
        payload["runs"][1]["baseline_metrics"] = None
        payload["runs"][1]["slicing"]["result"]["worst_slices"] = []
        payload["runs"][1]["hard_examples"]["result"]["summary"].pop("mean_uncertainty")
        payload["runs"][1]["hard_examples"]["result"]["summary"].pop("wrong_confident")
        payload["runs"][1]["robustness"] = {
            "status": "insufficient_data",
            "detail": "No numerical features were perturbed.",
            "result": None,
        }
        comparison = build_model_comparison(payload)["comparisons"][0]
        metrics = {item["name"]: item for item in comparison["metrics"]}
        tree = {row["model"]: row for row in metrics["baseline_f1"]["models"]}["tree"]
        self.assertIsNone(tree["value"])
        self.assertEqual(tree["status"], "insufficient_data")
        slice_row = {row["model"]: row for row in metrics["worst_slice_delta_f1"]["models"]}["tree"]
        self.assertIsNone(slice_row["value"])
        uncertainty = {row["model"]: row for row in metrics["mean_uncertainty"]["models"]}["tree"]
        self.assertEqual(uncertainty["status"], "insufficient_data")
        self.assertNotIn("most_uncertain", json.dumps(uncertainty))
        robustness = {row["model"]: row for row in metrics["robustness_f1_drop"]["models"]}["tree"]
        self.assertEqual(robustness["levels"], [])
        for finding in comparison["findings"].values():
            self.assertFalse(finding["identified"])
            self.assertEqual(finding["models"], [])

    def test_unsupported_importance_does_not_stop_the_other_metrics(self):
        comparison = build_model_comparison(_payload())["comparisons"][0]
        metrics = {item["name"]: item for item in comparison["metrics"]}
        self.assertEqual(metrics["feature_importance"]["models"][0]["status"], "not_supported")
        self.assertEqual(metrics["baseline_f1"]["models"][0]["status"], "supported")
        self.assertTrue(comparison["findings"]["strongest_baseline_model"]["identified"])

    def test_datasets_are_not_pooled(self):
        payload = _payload()
        other = _run("other_data", "linear", f1=0.2, delta=-0.9, errors=9, uncertainty=0.9, drop=0.8)
        payload["runs"].append(other)
        comparisons = build_model_comparison(payload)["comparisons"]
        self.assertEqual([item["dataset"] for item in comparisons], ["sample_data", "other_data"])
        self.assertFalse(comparisons[1]["findings"]["strongest_baseline_model"]["identified"])

    def test_saved_json_keeps_exact_values(self):
        payload = _payload()
        payload["runs"][0]["baseline_metrics"]["f1"] = 0.123456789
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model_comparison(payload, Path(tmp) / "model_comparison.json")
            stored = json.loads(path.read_text(encoding="utf-8"))
            written = write_diagnostic_reports(payload, directory=tmp)
            self.assertEqual(path.name, "model_comparison.json")
            self.assertEqual(
                stored["comparisons"][0]["metrics"][0]["models"][0]["value"],
                0.123456789,
            )
            self.assertEqual(written.comparison_path.name, "model_comparison.json")
            report = json.loads(written.comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["comparisons"][0]["metrics"][0]["models"][0]["value"],
                0.123456789,
            )

    def test_comparison_does_not_recompute_metrics(self):
        with patch(
            "src.evaluation.evaluate.score_predictions",
            side_effect=AssertionError("recomputed"),
        ):
            comparison = build_model_comparison(_payload())
        self.assertEqual(len(comparison["comparisons"]), 1)


def _payload() -> dict:
    return {
        "datasets": ["sample_data"],
        "models": ["linear", "tree"],
        "random_state": 42,
        "test_size": 0.2,
        "runs": [
            _run("sample_data", "linear", f1=0.91, delta=-0.1, errors=1, uncertainty=0.2, drop=0.05),
            _run(
                "sample_data",
                "tree",
                f1=0.8,
                delta=-0.4,
                errors=4,
                uncertainty=0.55,
                drop=0.2,
                importance=True,
            ),
        ],
    }


def _run(dataset, model, *, f1, delta, errors, uncertainty, drop, importance=False) -> dict:
    return {
        "experiment": {
            "dataset": dataset,
            "model": model,
            "random_state": 42,
            "test_size": 0.2,
        },
        "baseline_metrics": {"f1": f1, "accuracy": f1},
        "slicing": {
            "status": "supported",
            "detail": None,
            "result": {
                "overall_metrics": {"f1": f1},
                "worst_slices": [
                    {
                        "feature": "width",
                        "slice_definition": "Q4: (60, 90]",
                        "sample_count": 20,
                        "delta_f1": delta,
                    }
                ],
            },
        },
        "hard_examples": {
            "status": "supported",
            "detail": None,
            "result": {
                "summary": {
                    "wrong_confident": errors,
                    "mean_uncertainty": uncertainty,
                    "n_samples": 20,
                },
                "high_confidence_errors": [{"sample_id": 1}] * min(errors, 2),
                "most_uncertain": [{"sample_id": 1, "uncertainty": 0.99}],
            },
        },
        "robustness": {
            "status": "supported",
            "detail": None,
            "result": {
                "results": [
                    {
                        "noise_level": 0.1,
                        "mean": {"f1": f1},
                        "std": {"f1": 0.0},
                        "degradation": {"f1": drop + 0.3},
                    },
                    {
                        "noise_level": 0.3,
                        "mean": {"f1": f1 - drop},
                        "std": {"f1": 0.01},
                        "degradation": {"f1": drop},
                    },
                ]
            },
        },
        "explanation": (
            {
                "status": "supported",
                "detail": None,
                "result": {
                    "features": [
                        {"feature": "width", "importance": 0.7, "rank": 1},
                        {"feature": "group", "importance": 0.3, "rank": 2},
                    ]
                },
            }
            if importance
            else {
                "status": "not_supported",
                "detail": "This model does not expose feature_importances_.",
                "result": {
                    "status": "not_supported",
                    "reason": "This model does not expose feature_importances_.",
                    "features": [],
                },
            }
        ),
    }
