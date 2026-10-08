"""Tests for Gaussian noise on numerical test features."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.robustness import (
    DEFAULT_N_SEEDS,
    DEFAULT_NOISE_LEVELS,
    compare_robustness,
    compare_robustness_files,
    perturb_features,
    robustness_directory,
    run_robustness,
    save_robustness_result,
)
from src.evaluation.evaluate import LABEL_METRIC_NAMES


class RobustnessTests(unittest.TestCase):
    def test_zero_noise_matches_the_baseline(self):
        frame, labels = _measured_frame()
        model = ThresholdModel()
        result = run_robustness(
            model,
            frame,
            labels,
            noise_levels=(0.0,),
            n_seeds=4,
            random_state=42,
        )
        level = result.results[0]
        for name in LABEL_METRIC_NAMES:
            self.assertEqual(level["mean"][name], result.baseline_metrics[name])
            self.assertEqual(level["std"][name], 0.0)
            self.assertEqual(level["degradation"][name], 0.0)
            self.assertTrue(
                all(run[name] == result.baseline_metrics[name] for run in level["runs"])
            )
        self.assertEqual(model.fit_calls, 0)
        self.assertEqual(model.predict_calls, 5)

    def test_non_numerical_columns_are_not_noised(self):
        frame, _labels = _measured_frame()
        for name in ("city", "flag"):
            with self.assertRaises(ValueError):
                perturb_features(
                    frame,
                    numeric_features=[name],
                    noise_level=0.5,
                    seed=1,
                )
        pd.testing.assert_series_equal(frame["flag"], _measured_frame()[0]["flag"])

    def test_categorical_columns_stay_unchanged(self):
        frame, _labels = _measured_frame()
        perturbed = perturb_features(
            frame,
            numeric_features=["measure"],
            noise_level=0.3,
            seed=5,
        )
        pd.testing.assert_series_equal(perturbed["city"], frame["city"])
        pd.testing.assert_series_equal(perturbed["flag"], frame["flag"])

    def test_selected_numerical_column_is_perturbed(self):
        frame, _labels = _measured_frame()
        perturbed = perturb_features(
            frame,
            numeric_features=["measure"],
            noise_level=0.2,
            seed=5,
        )
        self.assertFalse(perturbed["measure"].equals(frame["measure"]))
        pd.testing.assert_series_equal(perturbed["other"], frame["other"])

    def test_same_seed_repeats_the_perturbation(self):
        frame, labels = _measured_frame()
        first = perturb_features(frame, numeric_features=["measure", "other"], noise_level=0.25, seed=11)
        second = perturb_features(frame, numeric_features=["measure", "other"], noise_level=0.25, seed=11)
        different = perturb_features(
            frame,
            numeric_features=["measure", "other"],
            noise_level=0.25,
            seed=12,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first["measure"].equals(different["measure"]))
        scored_first = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            noise_levels=(0.25,),
            seeds=(11, 12),
        )
        scored_second = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            noise_levels=(0.25,),
            seeds=(11, 12),
        )
        self.assertEqual(scored_first.to_dict(), scored_second.to_dict())

    def test_seeds_are_aggregated(self):
        self.assertEqual(DEFAULT_N_SEEDS, 5)
        self.assertEqual(DEFAULT_NOISE_LEVELS, (0.0, 0.05, 0.10, 0.20, 0.30))
        frame, labels = _measured_frame()
        result = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            noise_levels=(0.3,),
            n_seeds=5,
            random_state=42,
        )
        self.assertEqual(result.noise_configuration["seeds"], (42, 43, 44, 45, 46))
        level = result.results[0]
        self.assertEqual(len(level["runs"]), 5)
        for name in LABEL_METRIC_NAMES:
            values = [run[name] for run in level["runs"]]
            self.assertAlmostEqual(level["mean"][name], float(np.mean(values)))
            self.assertAlmostEqual(level["std"][name], float(np.std(values, ddof=1)))

    def test_degradation_is_baseline_minus_mean(self):
        frame, labels = _measured_frame()
        result = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            dataset="toy",
            model_name="threshold",
            noise_levels=(3.0,),
            seeds=(1, 2, 3, 4, 5),
        )
        level = result.results[0]
        for name in LABEL_METRIC_NAMES:
            self.assertAlmostEqual(
                level["degradation"][name],
                result.baseline_metrics[name] - level["mean"][name],
            )
        self.assertGreater(level["degradation"]["accuracy"], 0.0)
        other = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            dataset="toy",
            model_name="other",
            noise_levels=(3.0,),
            seeds=(1,),
        )
        compared = compare_robustness((result, other))
        self.assertEqual(set(compared["model"]), {"threshold", "other"})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first_path = save_robustness_result(result, directory)
            second_path = save_robustness_result(other, directory)
            from_files = compare_robustness_files((first_path, second_path))
        pd.testing.assert_frame_equal(
            from_files.reset_index(drop=True),
            compared.reset_index(drop=True),
        )

    def test_empty_numerical_feature_set_keeps_the_baseline(self):
        frame, labels = _measured_frame()
        result = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            features=[],
            noise_levels=(0.0, 0.5),
            n_seeds=3,
        )
        for level in result.results:
            for name in LABEL_METRIC_NAMES:
                self.assertEqual(level["mean"][name], result.baseline_metrics[name])
                self.assertEqual(level["degradation"][name], 0.0)
        categorical = frame.loc[:, ["city"]].copy()
        constant = run_robustness(
            ConstantModel(),
            categorical,
            np.array([0, 1, 0, 1, 1, 0, 1, 0]),
            noise_levels=(0.2,),
            n_seeds=2,
        )
        self.assertEqual(constant.noise_configuration["perturbed_features"], ())
        self.assertEqual(
            constant.results[0]["mean"]["accuracy"],
            constant.baseline_metrics["accuracy"],
        )

    def test_missing_numerical_values_stay_missing(self):
        frame, _labels = _measured_frame()
        frame.loc[0, "measure"] = np.nan
        frame.loc[3, "measure"] = np.nan
        frame.loc[1, "city"] = None
        perturbed = perturb_features(
            frame,
            numeric_features=["measure"],
            noise_level=0.4,
            seed=9,
        )
        self.assertTrue(perturbed["measure"].isna().equals(frame["measure"].isna()))
        finite = frame["measure"].notna()
        self.assertFalse(
            np.allclose(
                perturbed.loc[finite, "measure"].to_numpy(dtype=float),
                frame.loc[finite, "measure"].to_numpy(dtype=float),
            )
        )
        pd.testing.assert_series_equal(perturbed["city"], frame["city"])

    def test_original_test_frame_is_not_mutated(self):
        frame, labels = _measured_frame()
        original = frame.copy(deep=True)
        run_robustness(
            ThresholdModel(),
            frame,
            labels,
            noise_levels=(0.0, 0.3),
            seeds=(3, 4),
            features=["measure"],
        )
        perturb_features(frame, numeric_features=["measure", "other"], noise_level=0.3, seed=3)
        pd.testing.assert_frame_equal(frame, original)

    def test_json_records_mean_std_and_degradation(self):
        frame, labels = _measured_frame()
        result = run_robustness(
            ThresholdModel(),
            frame,
            labels,
            dataset="toy",
            model_name="threshold",
            noise_levels=(0.0, 0.1),
            seeds=(2,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = robustness_directory(Path(tmp) / "outputs" / "experiments" / "run")
            path = save_robustness_result(result, directory)
            self.assertEqual(
                path,
                Path(tmp) / "outputs" / "diagnostics" / "robustness" / "toy_threshold_robustness.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"], "toy")
        self.assertEqual(payload["model"], "threshold")
        self.assertIn("baseline_metrics", payload)
        self.assertIn("noise_configuration", payload)
        self.assertEqual(payload["noise_configuration"]["perturbed_features"], ["measure", "other"])
        for level in payload["results"]:
            self.assertIn("mean", level)
            self.assertIn("std", level)
            self.assertIn("degradation", level)
            for name in LABEL_METRIC_NAMES:
                self.assertIn(name, level["mean"])
                self.assertIn(name, level["std"])
                self.assertIn(name, level["degradation"])


class ThresholdModel:
    def __init__(self):
        self.predict_calls = 0
        self.fit_calls = 0

    def fit(self, X, y=None):
        self.fit_calls += 1
        return self

    def predict(self, X):
        self.predict_calls += 1
        values = pd.to_numeric(X["measure"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return (values > 0.0).astype(int)


class ConstantModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)


def _measured_frame():
    frame = pd.DataFrame(
        {
            "measure": [-2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, -3.0],
            "other": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "city": ["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", None],
            "flag": [True, False, True, False, True, False, True, False],
        }
    )
    labels = (frame["measure"] > 0.0).astype(int).to_numpy()
    return frame, labels
