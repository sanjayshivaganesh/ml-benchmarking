"""Run one configured experiment and write its artifacts.

The command-line entry in ``main.py`` loads configuration, applies overrides,
and validates names before calling ``run_phase2``. This module then calls
``run_experiment`` and writes that run under ``outputs/experiments``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.diagnostics.hard_examples import hard_examples_directory
from src.evaluation.evaluate import METRIC_NAMES
from src.experiments.config import DiagnosticsConfig, ExperimentConfig
from src.experiments.experiment import ExperimentResult, run_experiment

_DISPLAY_DECIMALS = 4
STEP_LOADING = "Loading experiment"
STEP_GENERATING_PREDICTIONS = "Generating predictions"
STEP_REUSING_PREDICTIONS = "Reusing saved predictions"
STEP_SLICING = "Running error slicing"
STEP_HARD_EXAMPLES = "Mining hard examples"
STEP_ROBUSTNESS = "Running robustness analysis"
STEP_REPORT = "Generating diagnostic report"


@dataclass(frozen=True)
class Phase2Run:
    """One finished experiment and the directory that holds its files."""

    experiment_id: str
    output_dir: Path
    result: ExperimentResult | None = None
    diagnostics: Any = None
    report_paths: Any = None
    reused_artifacts: bool = False


def run_phase2(config: ExperimentConfig) -> Phase2Run:
    """Train and score the validated configuration.

    Artifacts are written to ``<output_dir>/<experiment_id>/``. The Phase 1
    metric, error, and report files are not used.
    """
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    output_dir = config.output_dir / experiment_id
    result = run_experiment(
        dataset=config.dataset,
        models=config.models,
        random_state=config.random_state,
        test_size=config.test_size,
        model_dir=output_dir,
    )
    if result.random_state != config.random_state or result.test_size != config.test_size:
        raise RuntimeError(
            "The experiment split does not match the configuration: "
            f"random_state {result.random_state}, test_size {result.test_size}."
        )
    _write_outputs(result, output_dir, experiment_id)
    return Phase2Run(
        experiment_id=experiment_id,
        output_dir=output_dir,
        result=result,
    )


def make_experiment_id(
    dataset: str,
    models: tuple[str, ...] | list[str],
    random_state: int,
    test_size: float,
) -> str:
    """Build a stable directory name from the dataset, models, and seed."""
    model_part = "+".join(models)
    return (
        f"{dataset}__{model_part}__rs{random_state}__ts{_format_test_size(test_size)}"
    )


def run_configured_experiment(
    config: ExperimentConfig,
    *,
    noise_levels: Sequence[float] | None = None,
    n_seeds: int | None = None,
    report_directory: Path | str | None = None,
    echo: Callable[[str], None] = print,
) -> Phase2Run:
    """Train or load one experiment, then run configured diagnostics.

    Diagnostics stay off unless ``config.diagnostics.enabled`` is true.
    A saved model, prediction file, and metric file are reused. Requesting
    diagnostics does not train that experiment again.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("run_configured_experiment expects an ExperimentConfig.")
    if not config.diagnostics.enabled:
        return run_phase2(config)
    if not callable(echo):
        raise TypeError("echo must be a callable.")
    from src.diagnostics.diagnostic_engine import run_diagnostics
    from src.diagnostics.report import write_diagnostic_reports
    from src.diagnostics.workflow import _has_saved_models

    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    progress = _Progress(_progress_total(config.diagnostics), echo)
    progress(STEP_LOADING)
    reused = _has_saved_models(experiment_dir, config.models)
    trained = None
    if reused:
        progress(STEP_REUSING_PREDICTIONS)
        _publish_saved_predictions(experiment_dir, config.models)
    else:
        progress(STEP_GENERATING_PREDICTIONS)
        trained = run_phase2(config)
    diagnosed = run_diagnostics(
        config,
        noise_levels=noise_levels,
        n_seeds=n_seeds,
        slicing=config.diagnostics.slicing,
        hard_examples=config.diagnostics.hard_examples,
        robustness=config.diagnostics.robustness,
        explanations=config.diagnostics.explanations,
        progress=progress,
    )
    report_paths = None
    if config.diagnostics.report:
        progress(STEP_REPORT)
        report_paths = write_diagnostic_reports(diagnosed, directory=report_directory)
    if config.diagnostics.plots:
        from src.diagnostics.plots import save_diagnostic_plots

        save_diagnostic_plots(diagnosed)
    return Phase2Run(
        experiment_id=experiment_id,
        output_dir=experiment_dir,
        result=None if trained is None else trained.result,
        diagnostics=diagnosed,
        report_paths=report_paths,
        reused_artifacts=reused,
    )


def format_summary(run: Phase2Run) -> str:
    """Return a short text summary of one finished experiment."""
    if run.result is None:
        lines = _saved_summary_lines(run)
    else:
        lines = _result_summary_lines(run)
    if run.report_paths is not None:
        lines.extend(
            [
                "",
                f"Report: {run.report_paths.markdown_path}",
                f"Robustness report: {run.report_paths.robustness_path}",
                f"Model comparison: {run.report_paths.comparison_path}",
            ]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the project-root command-line entry."""
    import main as cli

    return cli.main(argv)


def _result_summary_lines(run: Phase2Run) -> list[str]:
    result = run.result
    lines = [
        f"Experiment {run.experiment_id}",
        f"Dataset: {result.dataset}",
        f"Models: {', '.join(result.models)}",
        f"Class labels: {_class_label_text(result.class_labels)}",
        f"random_state: {result.random_state}",
        f"test_size: {result.test_size}",
        f"Output: {run.output_dir}",
        f"Hard examples: {hard_examples_directory(run.output_dir)}",
        "",
    ]
    for model_name in result.models:
        lines.append(_metric_line(model_name, result.metrics[model_name]))
    lines.extend(["", "Full-precision metrics are in metrics.json."])
    return lines


def _saved_summary_lines(run: Phase2Run) -> list[str]:
    metadata = json.loads((run.output_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run.output_dir / "metrics.json").read_text(encoding="utf-8"))
    class_labels = {
        int(label): name for label, name in dict(metadata.get("class_labels") or {}).items()
    }
    models = list(metadata.get("models") or [])
    lines = [
        f"Experiment {run.experiment_id}",
        f"Dataset: {metadata.get('dataset')}",
        f"Models: {', '.join(models)}",
        f"Class labels: {_class_label_text(class_labels)}",
        f"random_state: {metadata.get('random_state')}",
        f"test_size: {metadata.get('test_size')}",
        f"Output: {run.output_dir}",
        f"Hard examples: {hard_examples_directory(run.output_dir)}",
        "Reused saved predictions.",
        "",
    ]
    for model_name in models:
        lines.append(_metric_line(model_name, metrics[model_name]))
    lines.extend(["", "Full-precision metrics are in metrics.json."])
    return lines


def _metric_line(model_name: str, metrics: dict[str, Any]) -> str:
    rendered = "  ".join(
        f"{name}={metrics[name]:.{_DISPLAY_DECIMALS}f}" for name in METRIC_NAMES
    )
    return f"{model_name}: {rendered}"


def _publish_saved_predictions(experiment_dir: Path, models: tuple[str, ...] | list[str]) -> None:
    """Write prediction CSVs from ``predictions.json`` without scoring again."""
    from src.diagnostics.prediction_store import (
        PredictionStore,
        prediction_directory,
        prediction_filename,
    )

    payload = json.loads((experiment_dir / "predictions.json").read_text(encoding="utf-8"))
    destination = prediction_directory(experiment_dir)
    dataset = payload.get("dataset")
    for model_name in models:
        store = PredictionStore.from_phase2_predictions(payload, model_name)
        store.save_csv(destination / prediction_filename(str(dataset), model_name))


def _progress_total(diagnostics: DiagnosticsConfig) -> int:
    return 2 + int(diagnostics.slicing) + int(diagnostics.hard_examples) + int(
        diagnostics.robustness
    ) + int(diagnostics.report)


class _Progress:
    """Number the diagnostic stages in the order they run."""

    def __init__(self, total: int, echo: Callable[[str], None]) -> None:
        self.total = total
        self.echo = echo
        self.index = 0

    def __call__(self, label: str) -> None:
        self.index += 1
        self.echo(f"[{self.index}/{self.total}] {label}")


def _write_outputs(result: ExperimentResult, output_dir: Path, experiment_id: str) -> None:
    metadata = dict(result.metadata)
    metadata["experiment_id"] = experiment_id
    metadata["output_dir"] = str(output_dir)
    metadata["class_labels"] = {
        str(int(label)): str(name) for label, name in result.class_labels.items()
    }
    metadata["positive_class"] = _positive_class(result.class_labels)
    metadata["files"] = {
        "metrics": "metrics.json",
        "error_analysis": "error_analysis.json",
        "predictions": "predictions.json",
        "models": dict(metadata["filenames"]),
    }
    predictions = {
        "dataset": result.dataset,
        "test_index": [_json_index(value) for value in result.y_test.index],
        "y_test": [int(value) for value in result.y_test.tolist()],
        "models": {
            model_name: {
                "predictions": [
                    int(value) for value in result.fitted[model_name].predictions.tolist()
                ],
                "positive_class_probabilities": [
                    float(value)
                    for value in result.fitted[model_name].positive_class_probabilities.tolist()
                ],
            }
            for model_name in result.models
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "metrics.json", result.metrics)
    _write_json(output_dir / "error_analysis.json", result.error_analysis)
    _write_json(output_dir / "predictions.json", predictions)


def _class_label_text(class_labels: dict[Any, Any]) -> str:
    keys = sorted(class_labels, key=lambda key: int(key))
    return ", ".join(f"{int(key)} = {class_labels[key]}" for key in keys)


def _positive_class(class_labels: dict[Any, Any]) -> dict[str, Any]:
    matches = [key for key in class_labels if int(key) == 1]
    if len(matches) != 1:
        raise ValueError(
            "Experiment metadata needs the name of class 1. "
            f"Found labels {dict(class_labels)}."
        )
    return {"label": 1, "name": str(class_labels[matches[0]])}


def _json_index(value: Any) -> int | str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_test_size(test_size: float) -> str:
    text = format(float(test_size), ".10f").rstrip("0").rstrip(".")
    return text.replace(".", "p")


if __name__ == "__main__":
    import sys

    sys.exit(main())
