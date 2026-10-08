"""Tests for the experiment registry and configuration file."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.data_loader import SUPPORTED_DATASETS
from src.experiments.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    load_config,
)
from src.experiments.registry import (
    list_datasets,
    list_models,
    validate_dataset,
    validate_models,
)
from src.models.models import MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_datasets_come_from_the_phase1_loader(self):
        self.assertIs(list_datasets(), SUPPORTED_DATASETS)
        self.assertEqual(list_datasets(), ("breast_cancer", "adult"))
        self.assertEqual(validate_dataset("breast_cancer"), "breast_cancer")
        self.assertEqual(validate_dataset("adult"), "adult")

    def test_invalid_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset("iris")
        with self.assertRaises(TypeError):
            validate_dataset(None)

    def test_models_come_from_the_phase1_model_zoo(self):
        self.assertIs(list_models(), MODEL_NAMES)
        self.assertEqual(
            validate_models(["gradient_boosting", "logistic_regression"]),
            ("gradient_boosting", "logistic_regression"),
        )
        self.assertEqual(validate_models("random_forest"), ("random_forest",))

    def test_invalid_model_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_models(["logistic_regression", "support_vector_machine"])
        with self.assertRaises(ValueError):
            validate_models([])
        with self.assertRaises(ValueError):
            validate_models(["logistic_regression", "logistic_regression"])
        with self.assertRaises(TypeError):
            validate_models(None)


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads_the_phase1_combination(self):
        config = load_config()
        self.assertEqual(DEFAULT_CONFIG_PATH, PROJECT_ROOT / "config.json")
        self.assertEqual(config.dataset, "breast_cancer")
        self.assertEqual(
            config.models,
            ("logistic_regression", "random_forest", "gradient_boosting"),
        )
        self.assertEqual(config.random_state, DEFAULT_RANDOM_STATE)
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.test_size, DEFAULT_TEST_SIZE)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.output_dir, (PROJECT_ROOT / "outputs" / "experiments").resolve())

    def test_omitted_seed_and_split_use_the_phase1_defaults(self):
        payload = {
            "dataset": "adult",
            "models": ["random_forest"],
            "output_dir": "outputs/experiments/adult_forest",
        }
        config = load_config(self._write(payload))
        self.assertEqual(config.dataset, "adult")
        self.assertEqual(config.models, ("random_forest",))
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.test_size, 0.2)

    def test_explicit_seed_and_split_override_the_defaults(self):
        payload = {
            "dataset": "breast_cancer",
            "models": ["logistic_regression"],
            "random_state": 7,
            "test_size": 0.3,
            "output_dir": "outputs/experiments/custom",
        }
        config = load_config(self._write(payload))
        self.assertEqual(config.random_state, 7)
        self.assertEqual(config.test_size, 0.3)
        self.assertEqual(
            config.output_dir,
            (PROJECT_ROOT / "outputs" / "experiments" / "custom").resolve(),
        )

    def test_invalid_dataset_model_and_fields_are_rejected(self):
        valid = {
            "dataset": "breast_cancer",
            "models": ["logistic_regression"],
            "random_state": 42,
            "test_size": 0.2,
            "output_dir": "outputs/experiments",
        }
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "dataset": "iris"}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "models": ["logistic_regression", "svm"]}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "test_size": 1}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "random_state": 1.5}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "C": 0.1}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "output_dir": "outputs/metrics"}))
        with self.assertRaises(ValueError):
            load_config(self._write({**valid, "output_dir": "models"}))
        with self.assertRaises(ValueError):
            load_config(self._write({key: value for key, value in valid.items() if key != "dataset"}))

    def test_config_loading_is_not_in_the_experiment_runner(self):
        import src.experiments.experiment as experiment_module

        self.assertNotIn("load_config", experiment_module.__dict__)
        self.assertNotIn("json", experiment_module.__dict__)

    def _write(self, payload: dict) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump(payload, handle)
            self.addCleanup(Path(handle.name).unlink, missing_ok=True)
            return handle.name
