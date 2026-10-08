"""Tests for tables and plots built from an experiment result."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.experiments.visualization import (
    metric_comparison_table,
    plot_confusion_matrices,
    plot_confusion_matrix,
    plot_roc_curves,
)


def _result(
    models,
    metrics,
    y_test,
    probabilities,
    matrices,
    class_labels,
    dataset="toy",
):
    fitted = {
        name: SimpleNamespace(
            positive_class_probabilities=None
            if probabilities.get(name) is None
            else np.asarray(probabilities[name], dtype=float)
        )
        for name in models
    }
    error_analysis = {}
    for name in models:
        matrix = matrices.get(name)
        if matrix is None:
            continue
        error_analysis[name] = {
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": matrix,
        }
    return SimpleNamespace(
        dataset=dataset,
        models=tuple(models),
        class_labels=class_labels,
        metrics=metrics,
        error_analysis=error_analysis,
        fitted=fitted,
        y_test=np.asarray(y_test),
    )


class MetricTableTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_one_model_uses_the_stored_metric_values(self):
        scores = {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.123456789,
            "roc_auc": 0.66,
            "brier_score": 0.11,
        }
        table = metric_comparison_table(
            _result(
                ["logistic_regression"],
                {"logistic_regression": scores},
                [0, 1],
                {"logistic_regression": [0.2, 0.8]},
                {"logistic_regression": [[1, 0], [0, 1]]},
                {0: "no", 1: "yes"},
            )
        )
        self.assertEqual(list(table["model"]), ["logistic_regression"])
        for name, value in scores.items():
            self.assertEqual(table.loc[0, name], value)

    def test_multiple_models_keep_selection_order_and_exact_values(self):
        result = _result(
            ["random_forest", "logistic_regression"],
            {
                "random_forest": {
                    "accuracy": 0.5,
                    "precision": 0.4,
                    "recall": 0.3,
                    "f1": 0.2,
                    "roc_auc": 0.6,
                    "brier_score": 0.25,
                },
                "logistic_regression": {
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.65,
                    "f1": 0.6,
                    "roc_auc": 0.8,
                },
            },
            [0, 1, 1, 0],
            {
                "random_forest": [0.1, 0.9, 0.8, 0.2],
                "logistic_regression": [0.2, 0.7, 0.6, 0.3],
            },
            {
                "random_forest": [[2, 1], [1, 2]],
                "logistic_regression": [[3, 0], [1, 2]],
            },
            {0: "a", 1: "b"},
        )
        table = metric_comparison_table(result)
        self.assertEqual(list(table["model"]), ["random_forest", "logistic_regression"])
        self.assertEqual(table.loc[0, "f1"], 0.2)
        self.assertEqual(table.loc[1, "roc_auc"], 0.8)
        self.assertTrue(pd.isna(table.loc[1, "brier_score"]))

    def test_missing_metrics_raise_a_clear_error(self):
        result = _result(
            ["logistic_regression"],
            {},
            [0, 1],
            {"logistic_regression": [0.2, 0.8]},
            {"logistic_regression": [[1, 0], [0, 1]]},
            {0: "no", 1: "yes"},
        )
        with self.assertRaises(ValueError) as caught:
            metric_comparison_table(result)
        self.assertIn("logistic_regression", str(caught.exception))


class RocPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_one_model_curve_uses_stored_labels_probabilities_and_auc(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        result = _result(
            ["logistic_regression"],
            {"logistic_regression": {"roc_auc": 0.42}},
            labels,
            {"logistic_regression": probabilities},
            {"logistic_regression": [[1, 1], [0, 2]]},
            {0: "malignant", 1: "benign"},
        )
        ax = plot_roc_curves(result)
        expected_fpr, expected_tpr, _ = roc_curve(labels, probabilities, pos_label=1)
        model_line = ax.get_lines()[0]
        np.testing.assert_allclose(model_line.get_xdata(), expected_fpr)
        np.testing.assert_allclose(model_line.get_ydata(), expected_tpr)
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "logistic_regression (ROC-AUC 0.4200)")
        self.assertIn("toy", ax.get_title())

    def test_multiple_models_share_one_axes(self):
        labels = [0, 0, 1, 1]
        result = _result(
            ["logistic_regression", "random_forest"],
            {
                "logistic_regression": {"roc_auc": 0.75},
                "random_forest": {"roc_auc": 1.0},
            },
            labels,
            {
                "logistic_regression": [0.2, 0.4, 0.6, 0.8],
                "random_forest": [0.1, 0.2, 0.8, 0.9],
            },
            {
                "logistic_regression": [[2, 0], [0, 2]],
                "random_forest": [[2, 0], [0, 2]],
            },
            {"0": "<=50K", "1": ">50K"},
        )
        figure, ax = plt.subplots()
        returned = plot_roc_curves(result, ax=ax)
        self.assertIs(returned, ax)
        self.assertEqual(len(ax.get_lines()), 3)
        legend = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(
            legend,
            [
                "logistic_regression (ROC-AUC 0.7500)",
                "random_forest (ROC-AUC 1.0000)",
                "chance",
            ],
        )
        plt.close(figure)

    def test_missing_probabilities_are_rejected(self):
        result = _result(
            ["logistic_regression"],
            {"logistic_regression": {"roc_auc": 0.5}},
            [0, 1],
            {"logistic_regression": None},
            {"logistic_regression": [[1, 0], [0, 1]]},
            {0: "no", 1: "yes"},
        )
        with self.assertRaises(ValueError) as caught:
            plot_roc_curves(result)
        self.assertIn("probabilities", str(caught.exception))


class ConfusionMatrixPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_matrix_uses_stored_counts_and_class_labels(self):
        cases = (
            ({0: "malignant", 1: "benign"}, ["0 = malignant", "1 = benign"]),
            ({"0": "<=50K", "1": ">50K"}, ["0 = <=50K", "1 = >50K"]),
        )
        for class_labels, expected in cases:
            with self.subTest(class_labels=class_labels):
                result = _result(
                    ["logistic_regression"],
                    {"logistic_regression": {"accuracy": 1.0}},
                    [0, 0, 1, 1],
                    {"logistic_regression": [0.1, 0.2, 0.8, 0.9]},
                    {"logistic_regression": [[4, 1], [2, 3]]},
                    class_labels,
                )
                ax = plot_confusion_matrix(result, "logistic_regression")
                self.assertEqual(
                    [tick.get_text() for tick in ax.get_xticklabels()],
                    expected,
                )
                self.assertEqual(
                    [tick.get_text() for tick in ax.get_yticklabels()],
                    expected,
                )
                self.assertEqual(
                    [text.get_text() for text in ax.texts],
                    ["4", "1", "2", "3"],
                )
                self.assertEqual(ax.get_xlabel(), "Predicted")
                self.assertEqual(ax.get_ylabel(), "True")
                plt.close(ax.figure)

    def test_each_selected_model_gets_its_own_matrix(self):
        result = _result(
            ["logistic_regression", "random_forest"],
            {
                "logistic_regression": {"accuracy": 0.5},
                "random_forest": {"accuracy": 0.5},
            },
            [0, 1],
            {
                "logistic_regression": [0.2, 0.8],
                "random_forest": [0.4, 0.6],
            },
            {
                "logistic_regression": [[1, 0], [0, 1]],
                "random_forest": [[0, 1], [1, 0]],
            },
            {0: "malignant", 1: "benign"},
        )
        figure = plot_confusion_matrices(result)
        axes = figure.axes
        matrix_axes = [ax for ax in axes if ax.get_title()]
        self.assertEqual(
            [ax.get_title() for ax in matrix_axes],
            ["logistic_regression", "random_forest"],
        )
        self.assertEqual(
            [text.get_text() for text in matrix_axes[1].texts],
            ["0", "1", "1", "0"],
        )
        plt.close(figure)

    def test_missing_matrix_and_missing_class_label_are_rejected(self):
        missing_matrix = _result(
            ["logistic_regression"],
            {"logistic_regression": {"accuracy": 1.0}},
            [0, 1],
            {"logistic_regression": [0.2, 0.8]},
            {},
            {0: "no", 1: "yes"},
        )
        with self.assertRaises(ValueError) as caught:
            plot_confusion_matrix(missing_matrix, "logistic_regression")
        self.assertIn("Confusion matrix", str(caught.exception))

        missing_label = _result(
            ["logistic_regression"],
            {"logistic_regression": {"accuracy": 1.0}},
            [0, 1],
            {"logistic_regression": [0.2, 0.8]},
            {"logistic_regression": [[1, 0], [0, 1]]},
            {0: "only-zero"},
        )
        with self.assertRaises(ValueError) as caught:
            plot_confusion_matrix(missing_label, "logistic_regression")
        self.assertIn("Class label 1", str(caught.exception))

    def test_visualization_does_not_recompute_or_load_data(self):
        source = (
            Path(__file__).resolve().parents[1] / "src" / "experiments" / "visualization.py"
        ).read_text(encoding="utf-8")
        for name in (
            "load_dataset",
            "train_one",
            "train_all",
            "evaluate_model",
            "analyze_errors",
            "run_experiment",
            "malignant",
            "benign",
            "<=50K",
            ">50K",
        ):
            self.assertNotIn(name, source)
