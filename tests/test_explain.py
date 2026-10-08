"""Tests for built-in feature importance after preprocessing."""

import json
import tempfile
import unittest
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.data_loader import load_dataset
from src.diagnostics.explain import (
    explain_feature_importance,
    explanations_directory,
    save_feature_importance,
)


class FeatureImportanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        breast = load_dataset("breast_cancer", test_size=0.2, random_state=42)
        adult = load_dataset("adult", test_size=0.2, random_state=42)
        cls.breast_names = list(breast.feature_names)
        cls.breast_forest = _fit(
            breast,
            RandomForestClassifier(n_estimators=10, random_state=0),
            rows=40,
        )
        cls.breast_boosting = _fit(
            breast,
            GradientBoostingClassifier(n_estimators=5, random_state=0),
            rows=40,
        )
        cls.breast_logistic = _fit(
            breast,
            LogisticRegression(max_iter=5000),
            rows=40,
        )
        cls.adult_forest = _fit(
            adult,
            RandomForestClassifier(n_estimators=5, random_state=0),
            rows=80,
        )

    def test_tree_importances_match_the_fitted_classifier(self):
        for model_name, pipeline in (
            ("random_forest", self.breast_forest),
            ("gradient_boosting", self.breast_boosting),
        ):
            with self.subTest(model=model_name):
                result = explain_feature_importance(
                    pipeline,
                    dataset="breast_cancer",
                    model_name=model_name,
                )
                estimator = pipeline.named_steps["classifier"]
                names = list(pipeline.named_steps["preprocessing"].get_feature_names_out())
                expected = {
                    name: float(importance)
                    for name, importance in zip(names, estimator.feature_importances_)
                }
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.method, "feature_importances_")
                self.assertEqual(len(result.features), len(expected))
                for row in result.features:
                    self.assertEqual(row["importance"], expected[row["feature"]])

    def test_features_are_ranked_by_descending_importance(self):
        result = explain_feature_importance(
            self.breast_forest,
            dataset="breast_cancer",
            model_name="random_forest",
        )
        names = list(self.breast_forest.named_steps["preprocessing"].get_feature_names_out())
        importances = self.breast_forest.named_steps["classifier"].feature_importances_
        order = sorted(
            range(len(names)),
            key=lambda index: (-float(importances[index]), names[index]),
        )
        self.assertEqual(
            [row["feature"] for row in result.features],
            [names[index] for index in order],
        )
        self.assertEqual(
            [row["rank"] for row in result.features],
            list(range(1, len(names) + 1)),
        )
        importances_out = [row["importance"] for row in result.features]
        self.assertEqual(importances_out, sorted(importances_out, reverse=True))

    def test_names_match_the_preprocessed_model_inputs(self):
        breast = explain_feature_importance(
            self.breast_forest,
            dataset="breast_cancer",
            model_name="random_forest",
        )
        self.assertEqual(
            sorted(row["feature"] for row in breast.features),
            sorted(self.breast_names),
        )
        adult = explain_feature_importance(
            self.adult_forest,
            dataset="adult",
            model_name="random_forest",
        )
        transformed = list(self.adult_forest.named_steps["preprocessing"].get_feature_names_out())
        self.assertEqual(
            sorted(row["feature"] for row in adult.features),
            sorted(transformed),
        )
        self.assertIn("age", transformed)
        self.assertNotIn("workclass", transformed)
        self.assertTrue(any(name.startswith("workclass_") for name in transformed))

    def test_models_without_feature_importances_are_skipped(self):
        models = (
            ("random_forest", self.breast_forest),
            ("logistic_regression", self.breast_logistic),
        )
        results = []
        for model_name, pipeline in models:
            results.append(
                explain_feature_importance(
                    pipeline,
                    dataset="breast_cancer",
                    model_name=model_name,
                )
            )
        self.assertEqual(results[0].status, "ok")
        logistic = results[1]
        self.assertEqual(logistic.status, "not_supported")
        self.assertIsNone(logistic.method)
        self.assertEqual(logistic.features, ())
        self.assertIn("feature_importances_", logistic.reason)
        self.assertFalse(hasattr(self.breast_logistic.named_steps["classifier"], "feature_importances_"))

    def test_json_uses_the_explanation_directory(self):
        result = explain_feature_importance(
            self.breast_forest,
            dataset="breast_cancer",
            model_name="random_forest",
        )
        unsupported = explain_feature_importance(
            self.breast_logistic,
            dataset="breast_cancer",
            model_name="logistic_regression",
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = explanations_directory(Path(tmp) / "outputs" / "experiments" / "run")
            path = save_feature_importance(result, directory)
            skipped = save_feature_importance(unsupported, directory)
            self.assertEqual(
                path,
                Path(tmp)
                / "outputs"
                / "diagnostics"
                / "explanations"
                / "breast_cancer_random_forest_feature_importance.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            skipped_payload = json.loads(skipped.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["features"][0]["rank"], 1)
        self.assertIn("feature", payload["features"][0])
        self.assertIn("importance", payload["features"][0])
        self.assertEqual(skipped_payload["status"], "not_supported")
        self.assertEqual(skipped_payload["features"], [])


def _fit(split, estimator, rows: int):
    pipeline = Pipeline(
        steps=[
            ("preprocessing", clone(split.preprocessor)),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(split.X_train.iloc[:rows], split.y_train.iloc[:rows])
    return pipeline
