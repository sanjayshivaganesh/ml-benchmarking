"""Run one configured experiment and write its artifacts.

The command-line entry in ``main.py`` loads configuration, applies overrides,
and validates names before calling ``run_phase2``. This module then calls
``run_experiment`` and writes that run under ``outputs/experiments``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluate import METRIC_NAMES
from src.experiments.config import ExperimentConfig
from src.experiments.experiment import ExperimentResult, run_experiment

_DISPLAY_DECIMALS = 4


@dataclass(frozen=True)
class Phase2Run:
    """One finished experiment and the directory that holds its files."""

    experiment_id: str
    output_dir: Path
    result: ExperimentResult


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


def format_summary(run: Phase2Run) -> str:
    """Return a short text summary of one finished experiment."""
    result = run.result
    lines = [
        f"Experiment {run.experiment_id}",
        f"Dataset: {result.dataset}",
        f"Models: {', '.join(result.models)}",
        f"Class labels: {_class_label_text(result.class_labels)}",
        f"random_state: {result.random_state}",
        f"test_size: {result.test_size}",
        f"Output: {run.output_dir}",
        "",
    ]
    for model_name in result.models:
        metrics = result.metrics[model_name]
        rendered = "  ".join(
            f"{name}={metrics[name]:.{_DISPLAY_DECIMALS}f}" for name in METRIC_NAMES
        )
        lines.append(f"{model_name}: {rendered}")
    lines.extend(["", "Full-precision metrics are in metrics.json."])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the project-root command-line entry."""
    import main as cli

    return cli.main(argv)


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
