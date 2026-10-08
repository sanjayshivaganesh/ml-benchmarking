"""Tests for the Streamlit experiment dashboard."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt

import app
from src.data.data_loader import SUPPORTED_DATASETS
from src.diagnostics.hard_examples import hard_examples_directory
from src.experiments.visualization import (
    metric_comparison_table,
    plot_confusion_matrices,
    plot_roc_curves,
)
from src.models.models import MODEL_NAMES


class DashboardSelectionTests(unittest.TestCase):
    def test_selectors_come_from_the_registries(self):
        self.assertIs(app.dataset_choices(), SUPPORTED_DATASETS)
        self.assertIs(app.model_choices(), MODEL_NAMES)

    def test_sidebar_values_override_the_configuration(self):
        config = app.experiment_config(
            "adult",
            ["random_forest", "logistic_regression"],
            random_state=7,
            test_size=0.3,
        )
        self.assertEqual(config.dataset, "adult")
        self.assertEqual(config.models, ("random_forest", "logistic_regression"))
        self.assertEqual(config.random_state, 7)
        self.assertEqual(config.test_size, 0.3)

    def test_selection_reaches_experiment(self):
        seen = {}

        def capture(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop before training")

        with patch("src.run_phase2.run_experiment", side_effect=capture):
            with self.assertRaises(RuntimeError):
                app.run_dashboard(
                    "adult",
                    ["logistic_regression", "random_forest"],
                    random_state=7,
                    test_size=0.25,
                )
        self.assertEqual(seen["dataset"], "adult")
        self.assertEqual(seen["models"], ("logistic_regression", "random_forest"))
        self.assertEqual(seen["random_state"], 7)
        self.assertEqual(seen["test_size"], 0.25)

    def test_no_model_and_invalid_settings_are_readable(self):
        with self.assertRaises(ValueError) as caught:
            app.run_dashboard("breast_cancer", [])
        self.assertIn("No models were selected", str(caught.exception))
        self.assertNotIn("Traceback", app.format_error(caught.exception))

        with self.assertRaises(ValueError) as caught:
            app.experiment_config("not_a_dataset", ["logistic_regression"])
        message = app.format_error(caught.exception)
        self.assertIn("Available datasets", message)
        self.assertNotIn("Traceback", message)

        with self.assertRaises(ValueError) as caught:
            app.experiment_config("breast_cancer", ["logistic_regression"], test_size=1)
        self.assertIn("test_size", app.format_error(caught.exception))
        self.assertEqual(app.format_error(RuntimeError("disk full")), "disk full")
        self.assertEqual(
            app.format_error(OSError("disk full")),
            "Experiment failed: disk full",
        )

    def test_page_populates_selectors_from_the_registries(self):
        from streamlit.testing.v1 import AppTest

        path = Path(__file__).resolve().parents[1] / "app.py"
        page = AppTest.from_file(path, default_timeout=30).run()
        self.assertFalse(page.exception)
        self.assertEqual(tuple(page.selectbox[0].options), SUPPORTED_DATASETS)
        self.assertEqual(tuple(page.multiselect[0].options), MODEL_NAMES)
        self.assertEqual(page.number_input[0].value, 42)
        self.assertEqual(page.number_input[1].value, 0.2)
        body = " ".join(element.value for element in page.markdown)
        self.assertIn("classification experiment", body)
        self.assertEqual(page.button[0].label, "Run Experiment")

    def test_app_does_not_hardcode_labels_or_training(self):
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        for name in (
            "load_dataset",
            "train_one",
            "evaluate_model",
            "analyze_errors",
            "malignant",
            "benign",
            "<=50K",
            ">50K",
            *SUPPORTED_DATASETS,
            *MODEL_NAMES,
        ):
            self.assertNotIn(name, source)


class DashboardResultTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_class_labels_come_from_the_result(self):
        result = SimpleNamespace(class_labels={0: "<=50K", 1: ">50K"})
        self.assertEqual(app.class_label_lines(result), ["0 = <=50K", "1 = >50K"])

    def test_experiment_renders_metrics_roc_and_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = app.run_dashboard(
                "breast_cancer",
                ["logistic_regression"],
                random_state=42,
                test_size=0.2,
                output_dir=tmp,
            )
            result = outcome.result
            table = metric_comparison_table(result)
            roc_axis = plot_roc_curves(result)
            figure = plot_confusion_matrices(result)
            self.assertTrue((outcome.output_dir / "metrics.json").is_file())
            self.assertTrue((outcome.output_dir / "error_analysis.json").is_file())
            hard_path = (
                hard_examples_directory(outcome.output_dir)
                / "breast_cancer_logistic_regression_hard_examples.json"
            )
            self.assertTrue(hard_path.is_file())
            hard_payload = json.loads(hard_path.read_text(encoding="utf-8"))
            hard_table = app.hard_example_table(hard_payload["high_confidence_errors"])
            self.assertIn("sample_id", hard_table.columns)
            self.assertIn("y_true", hard_table.columns)
            self.assertIn("y_pred", hard_table.columns)
            self.assertIn("mean radius", hard_table.columns)
            self.assertEqual(
                hard_table["sample_id"].tolist(),
                [row["sample_id"] for row in hard_payload["high_confidence_errors"]],
            )

        self.assertEqual(result.dataset, "breast_cancer")
        self.assertEqual(result.models, ("logistic_regression",))
        self.assertEqual(result.random_state, 42)
        self.assertEqual(result.test_size, 0.2)
        self.assertEqual(app.class_label_lines(result), ["0 = malignant", "1 = benign"])
        self.assertEqual(list(table["model"]), ["logistic_regression"])
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"):
            self.assertEqual(table.loc[0, name], result.metrics["logistic_regression"][name])
        self.assertGreaterEqual(len(roc_axis.get_lines()), 2)
        self.assertEqual(figure.axes[0].get_title(), "logistic_regression")
        self.assertIn("breast_cancer", outcome.experiment_id)
