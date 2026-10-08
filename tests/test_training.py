"""Tests for training every baseline model on every dataset."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.training import train as train_module
from src.training.train import train_all

EXPECTED_FILENAMES = (
    "breast_cancer__logistic_regression.joblib",
    "breast_cancer__random_forest.joblib",
    "breast_cancer__gradient_boosting.joblib",
    "adult__logistic_regression.joblib",
    "adult__random_forest.joblib",
    "adult__gradient_boosting.joblib",
)


class TrainAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls.tmp.name)
        cls.snapshots = []
        original_load = train_module.load_dataset

        def spy_load(*args, **kwargs):
            split = original_load(*args, **kwargs)
            cls.snapshots.append((split, split.y_test.copy()))
            return split

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with patch.object(train_module, "load_dataset", spy_load):
                cls.trained = train_all(model_dir=cls.model_dir, random_state=42)
        cls.convergence_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_six_combinations_train(self):
        combinations = [(item.dataset, item.model) for item in self.trained]
        self.assertEqual(
            combinations,
            [
                ("breast_cancer", "logistic_regression"),
                ("breast_cancer", "random_forest"),
                ("breast_cancer", "gradient_boosting"),
                ("adult", "logistic_regression"),
                ("adult", "random_forest"),
                ("adult", "gradient_boosting"),
            ],
        )
        self.assertEqual(len(self.snapshots), 6)

    def test_all_six_model_files_are_produced(self):
        produced = sorted(path.name for path in self.model_dir.glob("*.joblib"))
        self.assertEqual(produced, sorted(EXPECTED_FILENAMES))
        for trained in self.trained:
            self.assertTrue(trained.path.is_file())
            self.assertEqual(trained.path.name, f"{trained.dataset}__{trained.model}.joblib")

    def test_saved_models_load_and_predict_on_test_features(self):
        self.assertEqual(len(self.snapshots), len(self.trained))
        for trained, (split, _original_y_test) in zip(self.trained, self.snapshots):
            loaded = joblib.load(trained.path)
            self.assertIsInstance(loaded, Pipeline)
            check_is_fitted(loaded)
            predictions = loaded.predict(split.X_test)
            self.assertEqual(len(predictions), len(split.X_test))
            self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))
            self.assertEqual(loaded.metadata["dataset"], trained.dataset)
            self.assertEqual(loaded.metadata["model"], trained.model)
            self.assertEqual(loaded.metadata["random_state"], 42)
            self.assertEqual(loaded.metadata["filename"], trained.path.name)

    def test_logistic_regression_converges(self):
        self.assertEqual(self.convergence_warnings, [])

    def test_training_does_not_modify_test_labels(self):
        for split, original_y_test in self.snapshots:
            pd.testing.assert_series_equal(split.y_test, original_y_test)

    def test_saved_preprocessor_learns_from_training_rows_only(self):
        adult_pairs = [
            (trained, split)
            for trained, (split, _original) in zip(self.trained, self.snapshots)
            if trained.dataset == "adult"
        ]
        self.assertEqual(len(adult_pairs), 3)
        for trained, split in adult_pairs:
            loaded = joblib.load(trained.path)
            numeric = split.metadata["numeric_features"]
            scaler = (
                loaded.named_steps["preprocessing"]
                .named_transformers_["numeric"]
                .named_steps["scaler"]
            )
            train_median = split.X_train[numeric].median()
            imputed = split.X_train[numeric].fillna(train_median)
            np.testing.assert_allclose(
                scaler.mean_,
                imputed.mean().to_numpy(dtype=float),
            )
            combined = pd.concat(
                [split.X_train[numeric], split.X_test[numeric]],
                axis=0,
            )
            combined_mean = combined.fillna(train_median).mean().to_numpy(dtype=float)
            self.assertFalse(np.allclose(scaler.mean_, combined_mean))


class TrainAllInterfaceTests(unittest.TestCase):
    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            train_all(datasets=["iris"])


if __name__ == "__main__":
    unittest.main()
