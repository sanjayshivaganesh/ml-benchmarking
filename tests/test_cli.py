"""Tests for the Phase 2 command line and experiment outputs."""

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.config import apply_overrides, load_config
from src.experiments.registry import list_models, resolve_models
from src.models.models import MODEL_NAMES
from src.run_phase2 import make_experiment_id

import main as cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS
    }


class RegistrySelectionTests(unittest.TestCase):
    def test_all_resolves_from_the_model_registry(self):
        self.assertIs(resolve_models("all"), list_models())
        self.assertEqual(resolve_models(["all"]), MODEL_NAMES)
        self.assertNotIn("all", resolve_models(["all"]))

    def test_all_cannot_be_combined_with_named_models(self):
        with self.assertRaises(ValueError) as caught:
            resolve_models(["all", "logistic_regression"])
        self.assertIn("Available models", str(caught.exception))
        self.assertIn("random_forest", str(caught.exception))


class CommandLineTests(unittest.TestCase):
    def test_experiment_id_names_the_dataset_models_and_seed(self):
        self.assertEqual(
            make_experiment_id(
                "breast_cancer",
                ("logistic_regression", "random_forest"),
                42,
                0.2,
            ),
            "breast_cancer__logistic_regression+random_forest__rs42__ts0p2",
        )

    def test_cli_overrides_replace_config_values(self):
        config = load_config()
        updated = apply_overrides(
            config,
            dataset="adult",
            models=["all"],
            random_state=7,
            test_size=0.25,
        )
        self.assertEqual(updated.dataset, "adult")
        self.assertEqual(updated.models, MODEL_NAMES)
        self.assertEqual(updated.random_state, 7)
        self.assertEqual(updated.test_size, 0.25)
        self.assertEqual(updated.output_dir, config.output_dir)

    def test_invalid_dataset_lists_available_datasets(self):
        code, _stdout, stderr = self._run(["--dataset", "iris", "--models", "logistic_regression"])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
        self.assertIn("Available datasets", stderr)
        self.assertIn("breast_cancer", stderr)
        self.assertIn("adult", stderr)

    def test_invalid_model_lists_available_models(self):
        code, _stdout, stderr = self._run(
            ["--dataset", "breast_cancer", "--models", "support_vector_machine"]
        )
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
        self.assertIn("Available models", stderr)
        for name in MODEL_NAMES:
            self.assertIn(name, stderr)

    def test_no_models_lists_available_models(self):
        code, _stdout, stderr = self._run(["--dataset", "breast_cancer", "--models"])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
        self.assertIn("No models were selected", stderr)
        self.assertIn("gradient_boosting", stderr)

    def test_invalid_test_size_and_random_state_are_clear_errors(self):
        code, _stdout, stderr = self._run(["--test-size", "1"])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
        self.assertIn("test_size", stderr)

        from io import StringIO
        from contextlib import redirect_stderr

        captured = StringIO()
        with redirect_stderr(captured), self.assertRaises(SystemExit) as caught:
            cli.parse_args(["--random-state", "nope"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("invalid int value", captured.getvalue())
        self.assertNotIn("Traceback", captured.getvalue())

    def test_command_writes_an_isolated_experiment_directory(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "adult",
                        "models": ["gradient_boosting"],
                        "random_state": 42,
                        "test_size": 0.2,
                        "output_dir": tmp,
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = self._run(
                [
                    "--config",
                    str(config_path),
                    "--dataset",
                    "breast_cancer",
                    "--models",
                    "logistic_regression",
                    "random_forest",
                    "--test-size",
                    "0.3",
                    "--random-state",
                    "7",
                ]
            )
            experiment_dir = (
                Path(tmp)
                / "breast_cancer__logistic_regression+random_forest__rs7__ts0p3"
            )
            metrics = json.loads((experiment_dir / "metrics.json").read_text())
            metadata = json.loads((experiment_dir / "metadata.json").read_text())
            predictions = json.loads((experiment_dir / "predictions.json").read_text())
            saved_models = sorted(path.name for path in experiment_dir.glob("*.joblib"))
            self.assertTrue((experiment_dir / "error_analysis.json").is_file())

        self.assertEqual(code, 0, stderr)
        self.assertIn("Experiment breast_cancer__logistic_regression+random_forest__rs7__ts0p3", stdout)
        self.assertEqual(set(metrics), {"logistic_regression", "random_forest"})
        self.assertEqual(metadata["test_size"], 0.3)
        self.assertEqual(metadata["random_state"], 7)
        self.assertNotEqual(metadata["n_test_samples"], 114)
        self.assertEqual(len(predictions["y_test"]), metadata["n_test_samples"])
        self.assertEqual(
            saved_models,
            [
                "breast_cancer__logistic_regression.joblib",
                "breast_cancer__random_forest.joblib",
            ],
        )
        self.assertEqual(_artifact_bytes(), before)

    def test_entry_modules_do_not_hardcode_model_names(self):
        for relative in ("main.py", "src/run_phase2.py"):
            text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for name in MODEL_NAMES:
                self.assertNotIn(name, text)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        from io import StringIO
        from contextlib import redirect_stderr, redirect_stdout

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()
