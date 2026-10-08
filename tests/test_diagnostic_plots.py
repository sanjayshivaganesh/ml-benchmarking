"""Diagnostic plots are drawn from stored results and can be left off."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.diagnostics.plots import (
    plot_confidence_distribution,
    plot_feature_importance,
    plot_robustness_comparison,
    plot_robustness_curve,
    plot_slice_f1,
    save_diagnostic_plots,
)
from src.diagnostics.report import write_diagnostic_reports
from src.experiments.config import DiagnosticsConfig, ExperimentConfig
from src.run_phase2 import run_configured_experiment


class DiagnosticPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_slice_f1_uses_stored_slice_values(self):
        axis = plot_slice_f1(
            [
                {"feature": "width", "slice_definition": "Q1", "metrics": {"f1": 0.25}},
                {"feature": "width", "slice_definition": "Q2", "metrics": {"f1": 0.75}},
            ],
            dataset="sample_data",
            model="linear",
        )
        self.assertEqual([patch.get_width() for patch in axis.patches], [0.25, 0.75])
        self.assertEqual(axis.get_xlabel(), "F1")
        self.assertIn("sample_data / linear", axis.get_title())

    def test_robustness_curve_plots_noise_against_stored_f1(self):
        axis = plot_robustness_curve(
            [
                {"noise_level": 0.2, "mean": {"f1": 0.5}},
                {"noise_level": 0.0, "mean": {"f1": 0.9}},
            ]
        )
        line = axis.lines[0]
        self.assertEqual([float(value) for value in line.get_xdata()], [0.0, 0.2])
        self.assertEqual([float(value) for value in line.get_ydata()], [0.9, 0.5])
        self.assertEqual(axis.get_xlabel(), "Noise level")
        self.assertEqual(axis.get_ylabel(), "F1")

    def test_robustness_comparison_draws_one_line_per_model(self):
        axis = plot_robustness_comparison(
            [
                {"model": "linear", "results": [{"noise_level": 0.0, "mean": {"f1": 0.9}}]},
                {"model": "tree", "results": [{"noise_level": 0.0, "mean": {"f1": 0.7}}]},
            ],
            dataset="sample_data",
        )
        self.assertEqual([line.get_label() for line in axis.lines], ["linear", "tree"])
        self.assertEqual(axis.get_ylabel(), "F1")

    def test_feature_importance_keeps_the_stored_rank_order(self):
        axis = plot_feature_importance(
            [
                {"feature": "width", "importance": 0.7, "rank": 1},
                {"feature": "group", "importance": 0.3, "rank": 2},
            ]
        )
        self.assertEqual([tick.get_text() for tick in axis.get_yticklabels()], ["width", "group"])
        self.assertEqual([patch.get_width() for patch in axis.patches], [0.7, 0.3])

    def test_confidence_distribution_separates_correct_and_incorrect(self):
        axis = plot_confidence_distribution(
            pd.DataFrame(
                {
                    "confidence": [0.9, 0.8, 0.55],
                    "correct": [True, False, False],
                }
            )
        )
        self.assertEqual(
            [text.get_text() for text in axis.get_legend().get_texts()],
            ["correct", "incorrect"],
        )
        self.assertEqual(axis.get_xlabel(), "Confidence")

    def test_empty_measurements_are_not_plotted(self):
        with self.assertRaises(ValueError):
            plot_slice_f1([])
        with self.assertRaises(ValueError):
            plot_feature_importance([])
        with self.assertRaises(ValueError):
            plot_robustness_comparison(
                [{"model": "linear", "results": [{"noise_level": 0.0, "mean": {"f1": 0.9}}]}]
            )

    def test_saved_filenames_are_deterministic_and_skip_missing_data(self):
        predictions = {
            ("sample_data", model): [
                {"confidence": 0.9, "correct": True},
                {"confidence": 0.6, "correct": False},
            ]
            for model in ("linear", "tree")
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = save_diagnostic_plots(_result(), tmp, predictions=predictions)
            second = save_diagnostic_plots(_result(), tmp, predictions=predictions)
            names = sorted(path.name for path in first.values())
            self.assertEqual(
                names,
                [
                    "sample_data_linear_confidence_distribution.png",
                    "sample_data_linear_robustness_curve.png",
                    "sample_data_linear_slice_f1.png",
                    "sample_data_robustness_comparison.png",
                    "sample_data_tree_confidence_distribution.png",
                    "sample_data_tree_feature_importance.png",
                    "sample_data_tree_robustness_curve.png",
                    "sample_data_tree_slice_f1.png",
                ],
            )
            self.assertEqual(first, second)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in first.values()))

    def test_reports_are_written_when_plots_are_disabled(self):
        payload = {
            "datasets": ["sample_data"],
            "models": ["linear"],
            "random_state": 42,
            "test_size": 0.2,
            "runs": [
                {
                    "experiment": {"dataset": "sample_data", "model": "linear"},
                    "baseline_metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "src.diagnostics.plots.save_diagnostic_plots",
                side_effect=AssertionError("plotted"),
            ):
                written = write_diagnostic_reports(payload, directory=tmp)
            self.assertTrue(written.markdown_path.is_file())
            self.assertTrue(written.robustness_path.is_file())
            self.assertTrue(written.comparison_path.is_file())
            self.assertEqual(list(Path(tmp).glob("*.png")), [])

    def test_plot_generation_follows_the_diagnostics_flag(self):
        disabled = _config(plots=False)
        enabled = _config(plots=True)
        with _quiet_run() as plots:
            run_configured_experiment(disabled, echo=lambda _line: None)
        plots.assert_not_called()
        with _quiet_run() as plots:
            run_configured_experiment(enabled, echo=lambda _line: None)
        plots.assert_called_once()


class _quiet_run:
    def __enter__(self):
        self._patches = [
            patch("src.diagnostics.workflow._has_saved_models", return_value=False),
            patch("src.run_phase2.run_phase2", return_value=_phase2_run()),
            patch("src.diagnostics.diagnostic_engine.run_diagnostics", return_value={"runs": []}),
            patch("src.diagnostics.plots.save_diagnostic_plots") ,
        ]
        started = [item.start() for item in self._patches]
        self.plots = started[-1]
        return self.plots

    def __exit__(self, exc_type, exc, traceback):
        for item in reversed(self._patches):
            item.stop()
        return False


def _config(plots: bool) -> ExperimentConfig:
    return ExperimentConfig(
        dataset="breast_cancer",
        models=("logistic_regression",),
        random_state=42,
        test_size=0.2,
        output_dir=Path("unused"),
        diagnostics=DiagnosticsConfig(enabled=True, report=False, plots=plots),
    )


def _phase2_run():
    from src.run_phase2 import Phase2Run

    return Phase2Run("unused", Path("unused"), result=None)


def _result() -> dict:
    return {
        "runs": [
            _run("linear", importance=False),
            _run("tree", importance=True),
        ]
    }


def _run(model: str, importance: bool) -> dict:
    slices = [
        {
            "feature": "width",
            "slice_definition": "Q1",
            "metrics": {"f1": 0.4},
        }
    ]
    robustness = [{"noise_level": 0.0, "mean": {"f1": 0.8}}]
    explanation = (
        {
            "status": "supported",
            "result": {"features": [{"feature": "width", "importance": 0.7, "rank": 1}]},
        }
        if importance
        else {"status": "not_supported", "result": {"features": []}}
    )
    return {
        "experiment": {"dataset": "sample_data", "model": model},
        "slicing": {"status": "supported", "result": {"slices": slices}},
        "robustness": {"status": "supported", "result": {"results": robustness}},
        "explanation": explanation,
    }
