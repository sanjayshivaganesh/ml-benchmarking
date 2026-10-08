"""Tests for the Phase 1 baseline model registry."""

import unittest

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from src.models.models import get_models

EXPECTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


class GetModelsTests(unittest.TestCase):
    def test_returns_exactly_the_three_baseline_models(self):
        models = get_models()
        self.assertEqual(list(models), list(EXPECTED_MODELS))
        for name, estimator_type in EXPECTED_MODELS.items():
            self.assertIsInstance(models[name], estimator_type)

    def test_models_are_instantiated_and_unfitted(self):
        models = get_models()
        self.assertEqual(len(models), 3)
        for name, model in models.items():
            self.assertTrue(hasattr(model, "fit"))
            self.assertTrue(hasattr(model, "predict"))
            with self.subTest(model=name):
                with self.assertRaises(NotFittedError):
                    check_is_fitted(model)
        again = get_models()
        for name in EXPECTED_MODELS:
            self.assertIsNot(models[name], again[name])

    def test_random_state_is_deterministic(self):
        features, target = _toy_binary_problem()
        first = get_models(random_state=42)
        second = get_models(random_state=42)
        for name in EXPECTED_MODELS:
            self.assertEqual(first[name].get_params()["random_state"], 42)
            self.assertEqual(second[name].get_params()["random_state"], 42)
            first[name].fit(features, target)
            second[name].fit(features, target)
            np.testing.assert_array_equal(
                first[name].predict(features),
                second[name].predict(features),
            )

    def test_logistic_regression_max_iter_is_high_enough(self):
        model = get_models()["logistic_regression"]
        self.assertGreaterEqual(model.get_params()["max_iter"], 5000)

    def test_custom_random_state_is_forwarded(self):
        models = get_models(random_state=7)
        for name, model in models.items():
            with self.subTest(model=name):
                self.assertEqual(model.get_params()["random_state"], 7)


def _toy_binary_problem():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 4))
    target = (features[:, 0] + features[:, 1] > 0).astype(int)
    return features, target


if __name__ == "__main__":
    unittest.main()
