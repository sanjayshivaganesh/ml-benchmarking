"""Tests for Phase 1 Markdown report generation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis.report import (
    build_experiment_setup,
    describe_preprocessor,
    render_summary,
    run_phase1,
)
from src.evaluation.evaluate import METRIC_NAMES
from src.training.train import TrainedModel


def _scores(f1: float, roc_auc: float) -> dict[str, float]:
    return {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": 0.25,
    }


def _errors(
    false_positives: int,
    false_negatives: int,
    n_test: int = 100,
    feature_slices: dict | None = None,
) -> dict:
    count = false_positives + false_negatives
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "misclassified_count": count,
        "misclassification_rate": count / n_test,
        "n_test": n_test,
        "feature_slices": feature_slices or {},
    }


def _setup_for(metrics: dict) -> dict:
    datasets = {}
    models = {}
    for dataset_name, model_scores in metrics.items():
        datasets[dataset_name] = {
            "source": f"fixture:{dataset_name}",
            "class_labels": {0: "negative", 1: "positive"},
            "n_train": 80,
            "n_test": 100,
            "n_numeric_features": 2,
            "n_categorical_features": 1,
            "preprocessing": f"{dataset_name} preprocessing",
        }
        for model_name in model_scores:
            models.setdefault(
                model_name,
                {"random_state": 42},
            )
    return {
        "random_state": 42,
        "test_size": 0.2,
        "stratified": True,
        "sklearn_version": "test",
        "datasets": datasets,
        "models": models,
        "warnings": [],
    }


class RenderSummaryTests(unittest.TestCase):
    def test_report_contains_results_table_for_every_combination(self):
        metrics = {
            "breast_cancer": {
                "logistic_regression": _scores(0.90, 0.91),
                "random_forest": _scores(0.80, 0.95),
                "gradient_boosting": _scores(0.85, 0.88),
            },
            "adult": {
                "logistic_regression": _scores(0.60, 0.70),
                "random_forest": _scores(0.66, 0.72),
                "gradient_boosting": _scores(0.64, 0.75),
            },
        }
        error_report = {
            dataset: {
                model: _errors(false_positives=2, false_negatives=3)
                for model in models
            }
            for dataset, models in metrics.items()
        }
        summary = render_summary(metrics, error_report, _setup_for(metrics))
        self.assertTrue(summary.startswith("# ML Benchmarking — Phase 1\n"))
        self.assertIn(
            "| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |",
            summary,
        )
        for dataset_name, models in metrics.items():
            for model_name, scores in models.items():
                rendered = " | ".join(f"{scores[name]:.4f}" for name in METRIC_NAMES)
                self.assertIn(f"| {dataset_name} | {model_name} | {rendered} |", summary)
        self.assertEqual(summary.count("\n| breast_cancer |"), 6)
        self.assertEqual(summary.count("\n| adult |"), 6)

    def test_best_model_uses_f1_and_mentions_roc_auc(self):
        metrics = {
            "alpha": {
                "model_a": _scores(0.20, 0.60),
                "model_b": _scores(0.72, 0.55),
            }
        }
        error_report = {
            "alpha": {
                "model_a": _errors(1, 9),
                "model_b": _errors(1, 1),
            }
        }
        summary = render_summary(metrics, error_report, _setup_for(metrics))
        self.assertIn("`alpha` highest F1: `model_b` F1 0.7200, ROC-AUC 0.5500.", summary)
        self.assertIn("Highest ROC-AUC is `model_a`", summary)
        self.assertIn("ROC-AUC 0.6000", summary)

    def test_tied_f1_lists_every_winner(self):
        metrics = {
            "alpha": {
                "model_a": _scores(0.50, 0.40),
                "model_b": _scores(0.50, 0.70),
            }
        }
        error_report = {
            "alpha": {
                "model_a": _errors(1, 1),
                "model_b": _errors(1, 1),
            }
        }
        summary = render_summary(metrics, error_report, _setup_for(metrics))
        self.assertIn("`model_a` F1 0.5000, ROC-AUC 0.4000", summary)
        self.assertIn("`model_b` F1 0.5000, ROC-AUC 0.7000", summary)

    def test_failure_section_uses_error_counts_and_elevated_slices_only(self):
        metrics = {"alpha": {"model_a": _scores(0.5, 0.5), "model_b": _scores(0.4, 0.4)}}
        error_report = {
            "alpha": {
                "model_a": _errors(
                    false_positives=4,
                    false_negatives=12,
                    feature_slices={
                        "age": {
                            "type": "numeric",
                            "bins": [
                                {"bin": "(0, 10]", "count": 40, "errors": 2, "error_rate": 0.05},
                                {"bin": "(10, 20]", "count": 30, "errors": 12, "error_rate": 0.4},
                            ],
                        }
                    },
                ),
                "model_b": _errors(false_positives=1, false_negatives=1),
            }
        }
        summary = render_summary(metrics, error_report, _setup_for(metrics))
        self.assertIn("`model_a` misclassified the most alpha test rows: 16 of 100.", summary)
        self.assertIn("4 false positives and 12 false negatives", summary)
        self.assertIn("False negatives exceed false positives by 8.", summary)
        self.assertIn("`age` / `(10, 20]`", summary)
        self.assertNotIn("(0, 10]", summary)
        self.assertIn("No slice level met the elevation rule.", summary)

    def test_repeated_renders_match(self):
        metrics = {"alpha": {"model_a": _scores(0.2, 0.6), "model_b": _scores(0.72, 0.55)}}
        error_report = {
            "alpha": {"model_a": _errors(1, 9), "model_b": _errors(2, 2)}
        }
        setup = _setup_for(metrics)
        self.assertEqual(
            render_summary(metrics, error_report, setup),
            render_summary(metrics, error_report, setup),
        )

    def test_missing_metric_raises(self):
        metrics = {"alpha": {"model_a": {"accuracy": 0.5}}}
        error_report = {"alpha": {"model_a": _errors(1, 1)}}
        with self.assertRaises(ValueError):
            render_summary(metrics, error_report, _setup_for(metrics))


class PreprocessorDescriptionTests(unittest.TestCase):
    def test_passthrough_and_adult_style_steps_are_described_from_the_estimator(self):
        passthrough = ColumnTransformer(
            [("numeric", "passthrough", ["a", "b"])],
            remainder="drop",
        )
        passthrough.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        passthrough_text = describe_preprocessor(passthrough)
        self.assertIn("numeric (2 columns): passthrough", passthrough_text)

        adult_style = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    ["x"],
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    ["g"],
                ),
            ],
            remainder="drop",
        )
        adult_style.fit(pd.DataFrame({"x": [1.0, 2.0, None], "g": ["a", None, "b"]}))
        adult_text = describe_preprocessor(adult_style)
        self.assertIn("median imputation", adult_text)
        self.assertIn("standard scaling", adult_text)
        self.assertIn("most-frequent imputation", adult_text)
        self.assertIn("one-hot encoding", adult_text)


class BuildSetupTests(unittest.TestCase):
    def test_setup_comes_from_saved_pipeline_metadata(self):
        frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
        target = [0, 0, 1, 1]
        preprocessor = ColumnTransformer(
            [("numeric", "passthrough", ["a", "b"])],
            remainder="drop",
        )
        pipeline = Pipeline(
            [
                ("preprocessing", preprocessor),
                ("classifier", LogisticRegression(C=1.0, max_iter=200, random_state=42)),
            ]
        )
        pipeline.fit(frame, target)
        metadata = {
            "source": "fixture",
            "class_labels": {0: "no", 1: "yes"},
            "numeric_features": ["a", "b"],
            "categorical_features": [],
            "n_train_samples": 4,
            "random_state": 42,
            "test_size": 0.2,
            "stratified": True,
            "sklearn_version": "test",
            "hyperparameters": pipeline.named_steps["classifier"].get_params(),
        }
        pipeline.metadata = metadata
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy__logistic_regression.joblib"
            import joblib

            joblib.dump(pipeline, path)
            trained = [
                TrainedModel(
                    dataset="toy",
                    model="logistic_regression",
                    path=path,
                    metadata=metadata,
                )
            ]
            error_report = {"toy": {"logistic_regression": _errors(1, 0, n_test=4)}}
            setup = build_experiment_setup(trained, error_report, runtime_warnings=[])
        self.assertEqual(setup["random_state"], 42)
        self.assertEqual(setup["test_size"], 0.2)
        self.assertTrue(setup["stratified"])
        self.assertEqual(setup["datasets"]["toy"]["n_train"], 4)
        self.assertEqual(setup["datasets"]["toy"]["n_test"], 4)
        self.assertIn("passthrough", setup["datasets"]["toy"]["preprocessing"])
        self.assertEqual(setup["models"]["logistic_regression"]["max_iter"], 200)
        summary = render_summary(
            {"toy": {"logistic_regression": _scores(0.5, 0.5)}},
            error_report,
            setup,
        )
        self.assertIn("`toy` from `fixture`", summary)
        self.assertIn("0 = no, 1 = yes", summary)
        self.assertIn("passthrough", summary)


class RunPhase1OrchestrationTests(unittest.TestCase):
    def test_command_writes_summary_from_stage_outputs(self):
        metrics = {"alpha": {"model_a": _scores(0.72, 0.61)}}
        error_report = {"alpha": {"model_a": _errors(3, 5)}}
        setup = _setup_for(metrics)
        order = []

        def train_side_effect(**kwargs):
            order.append("train")
            return ["trained"]

        def evaluate_side_effect(**kwargs):
            order.append("evaluate")
            return metrics

        def analyze_side_effect(**kwargs):
            order.append("analyze")
            return error_report

        def setup_side_effect(*args, **kwargs):
            order.append("setup")
            return setup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.md"
            with (
                patch("src.analysis.report.train_all", side_effect=train_side_effect),
                patch("src.analysis.report.evaluate_all", side_effect=evaluate_side_effect),
                patch("src.analysis.report.analyze_all", side_effect=analyze_side_effect),
                patch(
                    "src.analysis.report.build_experiment_setup",
                    side_effect=setup_side_effect,
                ),
            ):
                result = run_phase1(
                    random_state=42,
                    model_dir=root / "models",
                    metrics_path=root / "metrics.json",
                    error_report_path=root / "error_report.json",
                    summary_path=summary_path,
                )
            text = summary_path.read_text(encoding="utf-8")
        self.assertEqual(order, ["train", "evaluate", "analyze", "setup"])
        self.assertEqual(result["summary"], text)
        self.assertIn("# ML Benchmarking — Phase 1", text)
        self.assertIn("F1 0.7200", text)
        json.dumps(metrics)


if __name__ == "__main__":
    unittest.main()
