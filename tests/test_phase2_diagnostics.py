"""Phase 2 runs diagnostics from the same configuration without retraining."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.config import DiagnosticsConfig, ExperimentConfig
from src.run_phase2 import (
    STEP_GENERATING_PREDICTIONS,
    STEP_HARD_EXAMPLES,
    STEP_LOADING,
    STEP_REPORT,
    STEP_REUSING_PREDICTIONS,
    STEP_ROBUSTNESS,
    STEP_SLICING,
    format_summary,
    run_configured_experiment,
    run_phase2,
)

import main as cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ARTIFACTS = (
    PROJECT_ROOT / "outputs" / "metrics" / "metrics.json",
    PROJECT_ROOT / "outputs" / "errors" / "error_report.json",
    PROJECT_ROOT / "outputs" / "reports" / "summary.md",
)


class Phase2DiagnosticOrchestratorTests(unittest.TestCase):
    def test_phase2_training_does_not_start_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), diagnostics=DiagnosticsConfig(enabled=True))
            with patch(
                "src.diagnostics.diagnostic_engine.run_diagnostics",
                side_effect=AssertionError("diagnosed"),
            ):
                outcome = run_phase2(config)
        self.assertIsNotNone(outcome.result)
        self.assertIsNone(outcome.diagnostics)
        self.assertFalse(outcome.reused_artifacts)

    def test_configured_run_reuses_artifacts_and_writes_the_report(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root / "outputs" / "experiments")
            messages = []
            first = run_configured_experiment(
                config,
                noise_levels=(0.0,),
                n_seeds=1,
                report_directory=root / "outputs" / "diagnostics",
                echo=messages.append,
            )
            metrics_bytes = (first.output_dir / "metrics.json").read_bytes()
            metrics = json.loads(metrics_bytes)
            diagnostic_root = root / "outputs" / "diagnostics"
            model = "logistic_regression"
            expected = {
                diagnostic_root / "slicing" / f"breast_cancer_{model}_slices.json",
                diagnostic_root / "hard_examples" / f"breast_cancer_{model}_hard_examples.json",
                diagnostic_root / "robustness" / f"breast_cancer_{model}_robustness.json",
                diagnostic_root / "explanations" / f"breast_cancer_{model}_feature_importance.json",
                diagnostic_root / "summaries" / f"breast_cancer_{model}_diagnostic_summary.json",
                diagnostic_root / "failure_analysis.md",
                diagnostic_root / "robustness_report.json",
                diagnostic_root / "model_comparison.json",
                root / "outputs" / "predictions" / f"breast_cancer_{model}_predictions.csv",
            }
            for path in expected:
                self.assertTrue(path.is_file(), path)
            self.assertTrue((first.output_dir / "predictions.json").is_file())
            self.assertEqual(
                messages,
                [
                    f"[1/6] {STEP_LOADING}",
                    f"[2/6] {STEP_GENERATING_PREDICTIONS}",
                    f"[3/6] {STEP_SLICING}",
                    f"[4/6] {STEP_HARD_EXAMPLES}",
                    f"[5/6] {STEP_ROBUSTNESS}",
                    f"[6/6] {STEP_REPORT}",
                ],
            )
            self.assertFalse(first.reused_artifacts)
            run = first.diagnostics.runs[0]
            self.assertTrue(run.experiment["reused_artifacts"])
            self.assertEqual(run.baseline_metrics["accuracy"], metrics[model]["accuracy"])
            self.assertEqual(run.baseline_metrics["f1"], metrics[model]["f1"])
            self.assertEqual(run.slicing.status, "supported")
            self.assertEqual(run.explanation.status, "not_supported")
            report = (diagnostic_root / "failure_analysis.md").read_text(encoding="utf-8")
            self.assertIn("# Model Failure Analysis", report)
            robustness = json.loads((diagnostic_root / "robustness_report.json").read_text())
            self.assertEqual(
                robustness["comparison"][0]["f1_mean"],
                run.robustness.result["results"][0]["mean"]["f1"],
            )
            prediction_csv = root / "outputs" / "predictions" / f"breast_cancer_{model}_predictions.csv"
            prediction_csv.unlink()
            again = []
            with patch(
                "src.experiments.experiment.train_one",
                side_effect=AssertionError("retrained"),
            ):
                second = run_configured_experiment(
                    config,
                    noise_levels=(0.0,),
                    n_seeds=1,
                    report_directory=diagnostic_root,
                    echo=again.append,
                )
            self.assertTrue(second.reused_artifacts)
            self.assertIsNone(second.result)
            self.assertEqual(again[1], f"[2/6] {STEP_REUSING_PREDICTIONS}")
            self.assertEqual((second.output_dir / "metrics.json").read_bytes(), metrics_bytes)
            self.assertTrue(prediction_csv.is_file())
            self.assertIn("Reused saved predictions.", format_summary(second))
            skipped = []
            with patch(
                "src.diagnostics.diagnostic_engine.run_robustness",
                side_effect=AssertionError("perturbed"),
            ):
                third = run_configured_experiment(
                    _config(
                        root / "outputs" / "experiments",
                        diagnostics=DiagnosticsConfig(enabled=True, robustness=False),
                    ),
                    noise_levels=(0.0,),
                    n_seeds=1,
                    report_directory=diagnostic_root,
                    echo=skipped.append,
                )
            self.assertEqual(
                skipped,
                [
                    f"[1/5] {STEP_LOADING}",
                    f"[2/5] {STEP_REUSING_PREDICTIONS}",
                    f"[3/5] {STEP_SLICING}",
                    f"[4/5] {STEP_HARD_EXAMPLES}",
                    f"[5/5] {STEP_REPORT}",
                ],
            )
            self.assertEqual(third.diagnostics.runs[0].robustness.status, "not_supported")
            self.assertIn("Disabled", third.diagnostics.runs[0].robustness.detail)
            self.assertEqual(third.diagnostics.runs[0].slicing.status, "supported")
        self.assertEqual(_artifact_bytes(), before)

    def test_cli_can_turn_diagnostics_off(self):
        before = _artifact_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "breast_cancer",
                        "models": ["logistic_regression"],
                        "random_state": 42,
                        "test_size": 0.2,
                        "output_dir": tmp,
                        "diagnostics": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = _run(
                ["--config", str(config_path), "--no-diagnostics"]
            )
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("[1/", stdout)
        self.assertNotIn("failure_analysis.md", stdout)
        self.assertIn("logistic_regression", stdout)
        self.assertEqual(_artifact_bytes(), before)


def _config(
    output_dir: Path,
    diagnostics: DiagnosticsConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        dataset="breast_cancer",
        models=("logistic_regression",),
        random_state=42,
        test_size=0.2,
        output_dir=output_dir,
        diagnostics=diagnostics if diagnostics is not None else DiagnosticsConfig(enabled=True),
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _artifact_bytes() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in PHASE1_ARTIFACTS}
