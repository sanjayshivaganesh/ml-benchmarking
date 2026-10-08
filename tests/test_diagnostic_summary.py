"""Associations between feature importance and low-performing slices."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.diagnostics.diagnostic_summary import (
    build_diagnostic_summary,
    save_diagnostic_summary,
    summary_directory,
)
from src.diagnostics.explain import FeatureImportanceResult
from src.diagnostics.slicing import ErrorSlicingResult


class DiagnosticSummaryTests(unittest.TestCase):
    def test_important_feature_with_a_weak_slice_is_associated(self):
        summary = build_diagnostic_summary(_importance(), _slicing())
        self.assertEqual(len(summary.associations), 1)
        row = summary.associations[0]
        self.assertEqual(row["feature"], "age")
        self.assertEqual(row["importance"], 0.24)
        self.assertEqual(row["importance_rank"], 1)
        self.assertEqual(row["worst_slice"], "Q4: (60, 90]")
        self.assertEqual(row["f1"], 0.42)
        self.assertEqual(row["overall_f1"], 0.71)
        self.assertEqual(row["delta_f1"], -0.29)
        self.assertTrue(row["substantial"])
        self.assertEqual(
            row["statement"],
            "age is among the model's most important features, "
            "while the slice Q4: (60, 90] exhibits substantial performance degradation.",
        )
        self.assertNotIn("causes", row["statement"].lower())
        self.assertTrue(summary.to_dict()["association_only"])
        self.assertIn("do not show that a feature caused", summary.to_dict()["note"])

    def test_encoded_columns_map_to_the_sliced_feature(self):
        importance = _importance(
            (
                {"feature": "city_a", "importance": 0.10, "rank": 2},
                {"feature": "city_b", "importance": 0.20, "rank": 1},
                {"feature": "age", "importance": 0.05, "rank": 3},
                {"feature": "not_a_slice", "importance": 0.50, "rank": 4},
            )
        )
        slicing = _slicing(
            slices=(
                _slice("city", "a", f1=0.40, delta=-0.31),
                _slice("city", "b", f1=0.70, delta=-0.01),
                _slice("age", "Q1: (18, 30]", f1=0.70, delta=-0.01),
            )
        )
        summary = build_diagnostic_summary(importance, slicing, top_features=1)
        self.assertEqual([row["feature"] for row in summary.associations], ["city"])
        city = summary.associations[0]
        self.assertAlmostEqual(city["importance"], 0.30)
        self.assertEqual(city["worst_slice"], "a")
        self.assertEqual(city["delta_f1"], -0.31)
        self.assertEqual(
            [item["feature"] for item in city["transformed_features"]],
            ["city_a", "city_b"],
        )

    def test_rank_limit_and_non_degraded_slices_are_excluded(self):
        features = tuple(
            {"feature": f"f{index}", "importance": 1.0 - index / 10, "rank": index}
            for index in range(1, 7)
        )
        slices = tuple(
            _slice(f"f{index}", "low", f1=0.20, delta=-0.50) for index in range(1, 7)
        )
        slices = slices + (_slice("f1", "high", f1=0.90, delta=0.10),)
        summary = build_diagnostic_summary(_importance(features), _slicing(slices=slices))
        self.assertEqual([row["feature"] for row in summary.associations], ["f1", "f2", "f3", "f4", "f5"])
        self.assertEqual(summary.associations[0]["worst_slice"], "low")
        improved = build_diagnostic_summary(
            _importance(({"feature": "age", "importance": 0.4, "rank": 1},)),
            _slicing(slices=(_slice("age", "Q4: (60, 90]", f1=0.90, delta=0.19),)),
        )
        self.assertEqual(improved.associations, ())

    def test_unsupported_importance_and_unreliable_slices_add_nothing(self):
        unsupported = build_diagnostic_summary(
            _importance(status="not_supported", features=()),
            _slicing(),
        )
        self.assertEqual(unsupported.importance_status, "not_supported")
        self.assertEqual(unsupported.associations, ())
        unreliable = _slice("age", "Q4: (60, 90]", f1=0.10, delta=None, reliable=False)
        summary = build_diagnostic_summary(
            _importance(),
            _slicing(slices=(unreliable,)),
        )
        self.assertEqual(summary.associations, ())

    def test_the_same_inputs_produce_the_same_summary(self):
        first = build_diagnostic_summary(_importance(), _slicing())
        second = build_diagnostic_summary(_importance(), _slicing())
        self.assertEqual(first.to_dict(), second.to_dict())
        mild = build_diagnostic_summary(
            _importance(),
            _slicing(slices=(_slice("age", "Q4: (60, 90]", f1=0.69, delta=-0.02),)),
        )
        self.assertFalse(mild.associations[0]["substantial"])
        self.assertIn("has lower F1 than the full test set", mild.associations[0]["statement"])

    def test_json_records_the_association_fields(self):
        summary = build_diagnostic_summary(_importance(), _slicing())
        with tempfile.TemporaryDirectory() as tmp:
            directory = summary_directory(Path(tmp) / "outputs" / "experiments" / "run")
            path = save_diagnostic_summary(summary, directory)
            self.assertEqual(
                path,
                Path(tmp)
                / "outputs"
                / "diagnostics"
                / "summaries"
                / "adult_random_forest_diagnostic_summary.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["overall_f1"], 0.71)
        association = payload["associations"][0]
        for key in ("feature", "importance", "worst_slice", "f1", "overall_f1", "delta_f1", "statement"):
            self.assertIn(key, association)
        self.assertNotIn("causes the model to fail", association["statement"])

    def test_different_datasets_are_rejected(self):
        other = _slicing()
        mismatched = ErrorSlicingResult(
            dataset="breast_cancer",
            model=other.model,
            overall_metrics=other.overall_metrics,
            min_slice_size=other.min_slice_size,
            numeric_bins=other.numeric_bins,
            slices=other.slices,
            table=other.table,
            worst_slices=other.worst_slices,
        )
        with self.assertRaises(ValueError):
            build_diagnostic_summary(_importance(), mismatched)


def _importance(features=None, status="ok"):
    if features is None:
        features = ({"feature": "age", "importance": 0.24, "rank": 1},)
    return FeatureImportanceResult(
        dataset="adult",
        model="random_forest",
        status=status,
        method="feature_importances_" if status == "ok" else None,
        reason=None if status == "ok" else "This model does not expose feature_importances_.",
        features=tuple(features),
    )


def _slicing(slices=None):
    if slices is None:
        slices = (_slice("age", "Q4: (60, 90]", f1=0.42, delta=-0.29),)
    return ErrorSlicingResult(
        dataset="adult",
        model="random_forest",
        overall_metrics={"accuracy": 0.8, "precision": 0.75, "recall": 0.7, "f1": 0.71},
        min_slice_size=20,
        numeric_bins=4,
        slices=tuple(slices),
        table=pd.DataFrame(),
        worst_slices=(),
    )


def _slice(feature, definition, *, f1, delta, reliable=True):
    return {
        "feature": feature,
        "feature_type": "numeric",
        "slice_definition": definition,
        "sample_count": 30,
        "reliable": reliable,
        "min_slice_size": 20,
        "metrics": {"accuracy": f1, "precision": f1, "recall": f1, "f1": f1},
        "delta_accuracy": delta,
        "delta_f1": delta,
    }
