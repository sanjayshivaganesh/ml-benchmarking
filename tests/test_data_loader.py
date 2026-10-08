"""Tests for programmatic dataset loading and leakage-safe preprocessing."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data.data_loader import load_dataset


class BreastCancerLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = load_dataset("breast_cancer")

    def test_split_sizes_are_nonzero(self):
        _assert_nonempty_split(self, self.split)

    def test_target_distribution_is_reasonable(self):
        _assert_reasonable_binary_distribution(self, self.split)

    def test_train_and_test_indices_do_not_overlap(self):
        _assert_disjoint_indices(self, self.split)

    def test_feature_names_are_preserved(self):
        self.assertEqual(len(self.split.feature_names), 30)
        self.assertIn("mean radius", self.split.feature_names)
        self.assertEqual(list(self.split.X_train.columns), self.split.feature_names)
        self.assertEqual(list(self.split.X_test.columns), self.split.feature_names)
        self.assertEqual(self.split.metadata["class_labels"][0], "malignant")
        self.assertEqual(self.split.metadata["class_labels"][1], "benign")

    def test_preprocessor_passes_features_through_without_fitting_on_test(self):
        with self.assertRaises(NotFittedError):
            check_is_fitted(self.split.preprocessor)

        preprocessor = clone(self.split.preprocessor)
        preprocessor.fit(self.split.X_train)
        transformed = preprocessor.transform(self.split.X_train)
        np.testing.assert_allclose(transformed, self.split.X_train.to_numpy())
        self.assertEqual(
            list(preprocessor.get_feature_names_out()),
            self.split.feature_names,
        )
        with self.assertRaises(NotFittedError):
            check_is_fitted(self.split.preprocessor)

    def test_repeated_calls_are_deterministic(self):
        _assert_deterministic_split(self, "breast_cancer")


class AdultLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = load_dataset("adult")

    def test_split_sizes_are_nonzero(self):
        _assert_nonempty_split(self, self.split)

    def test_target_distribution_is_reasonable(self):
        _assert_reasonable_binary_distribution(self, self.split)
        self.assertEqual(self.split.metadata["class_labels"], {0: "<=50K", 1: ">50K"})

    def test_train_and_test_indices_do_not_overlap(self):
        _assert_disjoint_indices(self, self.split)

    def test_preprocessing_handles_categorical_and_missing_values(self):
        split = self.split
        numeric = split.metadata["numeric_features"]
        categorical = split.metadata["categorical_features"]
        self.assertTrue(numeric)
        self.assertTrue(categorical)
        self.assertGreater(int(split.X_train.isna().sum().sum()), 0)
        self.assertGreater(int(split.X_test.isna().sum().sum()), 0)
        self.assertTrue(
            all(
                not pd.api.types.is_numeric_dtype(split.X_train[column])
                for column in categorical
            )
        )

        transformers = {
            name: transformer
            for name, transformer, _columns in split.preprocessor.transformers
        }
        numeric_pipeline = transformers["numeric"]
        categorical_pipeline = transformers["categorical"]
        self.assertIsInstance(numeric_pipeline, Pipeline)
        self.assertEqual(numeric_pipeline.named_steps["imputer"].strategy, "median")
        self.assertIsInstance(numeric_pipeline.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(categorical_pipeline, Pipeline)
        self.assertEqual(
            categorical_pipeline.named_steps["imputer"].strategy,
            "most_frequent",
        )
        self.assertIsInstance(
            categorical_pipeline.named_steps["encoder"],
            OneHotEncoder,
        )

        with self.assertRaises(NotFittedError):
            check_is_fitted(split.preprocessor)

        preprocessor = clone(split.preprocessor)
        preprocessor.fit(split.X_train)
        transformed_train = preprocessor.transform(split.X_train)
        transformed_test = preprocessor.transform(split.X_test)

        self.assertEqual(transformed_train.shape[0], len(split.X_train))
        self.assertEqual(transformed_test.shape[0], len(split.X_test))
        self.assertGreater(transformed_train.shape[1], split.X_train.shape[1])
        self.assertEqual(transformed_train.shape[1], transformed_test.shape[1])
        self.assertFalse(np.isnan(transformed_train).any())
        self.assertFalse(np.isnan(transformed_test).any())

        fitted_numeric = preprocessor.named_transformers_["numeric"]
        imputer = fitted_numeric.named_steps["imputer"]
        scaler = fitted_numeric.named_steps["scaler"]
        train_numeric = split.X_train[numeric]
        train_median = train_numeric.median()
        imputed = train_numeric.fillna(train_median)
        np.testing.assert_allclose(
            np.asarray(imputer.statistics_, dtype=float),
            train_median.to_numpy(dtype=float),
        )
        np.testing.assert_allclose(scaler.mean_, imputed.mean().to_numpy(dtype=float))
        np.testing.assert_allclose(
            scaler.scale_,
            imputed.std(ddof=0).to_numpy(dtype=float),
        )

        combined = pd.concat([split.X_train[numeric], split.X_test[numeric]], axis=0)
        combined_mean = combined.fillna(train_median).mean().to_numpy(dtype=float)
        self.assertFalse(np.allclose(scaler.mean_, combined_mean))

        cat_imputer = preprocessor.named_transformers_["categorical"].named_steps[
            "imputer"
        ]
        for column, learned in zip(categorical, cat_imputer.statistics_):
            mode = split.X_train[column].mode(dropna=True).iloc[0]
            self.assertEqual(learned, mode)

        with self.assertRaises(NotFittedError):
            check_is_fitted(split.preprocessor)

    def test_repeated_calls_are_deterministic(self):
        _assert_deterministic_split(self, "adult")


class LoadDatasetInterfaceTests(unittest.TestCase):
    def test_unknown_dataset_name_is_rejected(self):
        with self.assertRaises(ValueError):
            load_dataset("iris")


def _assert_nonempty_split(test_case, split):
    test_case.assertGreater(len(split.X_train), 0)
    test_case.assertGreater(len(split.X_test), 0)
    test_case.assertEqual(len(split.X_train), len(split.y_train))
    test_case.assertEqual(len(split.X_test), len(split.y_test))
    test_case.assertEqual(
        len(split.X_train) + len(split.X_test),
        split.metadata["n_samples"],
    )


def _assert_reasonable_binary_distribution(test_case, split):
    for part in (split.y_train, split.y_test):
        classes = set(part.astype(int).unique())
        test_case.assertEqual(classes, {0, 1})
        shares = part.value_counts(normalize=True)
        test_case.assertGreater(float(shares.min()), 0.05)
        test_case.assertLess(float(shares.max()), 0.95)
    train_positive_rate = float(split.y_train.mean())
    test_positive_rate = float(split.y_test.mean())
    test_case.assertLess(abs(train_positive_rate - test_positive_rate), 0.02)


def _assert_disjoint_indices(test_case, split):
    train_index = set(split.X_train.index)
    test_index = set(split.X_test.index)
    test_case.assertTrue(train_index.isdisjoint(test_index))
    test_case.assertEqual(
        len(train_index) + len(test_index),
        split.metadata["n_samples"],
    )
    test_case.assertTrue(split.X_train.index.equals(split.y_train.index))
    test_case.assertTrue(split.X_test.index.equals(split.y_test.index))


def _assert_deterministic_split(test_case, name):
    first = load_dataset(name, random_state=42)
    second = load_dataset(name, random_state=42)
    test_case.assertTrue(first.X_train.index.equals(second.X_train.index))
    test_case.assertTrue(first.X_test.index.equals(second.X_test.index))
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_train, second.y_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


if __name__ == "__main__":
    unittest.main()
