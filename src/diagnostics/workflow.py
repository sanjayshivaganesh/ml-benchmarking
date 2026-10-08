"""Run Phase 3 diagnostics for one configured experiment.

The dataset, models, seed, and output directory come from the Phase 2
configuration. When that experiment already has ``predictions.json``, slicing
and hard-example analysis use those saved predictions and do not train again.
Robustness testing loads the saved model artifact and the same test split,
then scores noisy copies of the test features. It does not fit the model.
The test features are reloaded with the same ``random_state`` and ``test_size``
so ``sample_id`` can be joined back to the original rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from src.data.data_loader import load_dataset
from src.diagnostics.hard_examples import (
    explore_hard_examples,
    hard_examples_directory,
    save_hard_examples,
)
from src.diagnostics.prediction_store import PredictionStore
from src.diagnostics.robustness import (
    DEFAULT_N_SEEDS,
    DEFAULT_NOISE_LEVELS,
    compare_robustness_files,
    robustness_directory,
    run_robustness,
    save_robustness_result,
)
from src.diagnostics.slicing import run_error_slicing, save_slicing_result, slicing_directory
from src.experiments.config import ExperimentConfig
from src.run_phase2 import make_experiment_id, run_phase2


@dataclass(frozen=True)
class DiagnosticSlicing:
    """Slice files written for one experiment."""

    experiment_id: str
    experiment_dir: Path
    dataset: str
    models: tuple[str, ...]
    output_paths: dict[str, Path]
    reused_predictions: bool


@dataclass(frozen=True)
class DiagnosticHardExamples:
    """Hard-example files written for one experiment."""

    experiment_id: str
    experiment_dir: Path
    dataset: str
    models: tuple[str, ...]
    output_paths: dict[str, Path]
    reused_predictions: bool


@dataclass(frozen=True)
class DiagnosticRobustness:
    """Robustness files written for one experiment."""

    experiment_id: str
    experiment_dir: Path
    dataset: str
    models: tuple[str, ...]
    output_paths: dict[str, Path]
    reused_model: bool
    comparison: pd.DataFrame


@dataclass(frozen=True)
class DiagnosticReport:
    """Slice files and hard-example files for one experiment."""

    experiment_id: str
    experiment_dir: Path
    dataset: str
    models: tuple[str, ...]
    slice_paths: dict[str, Path]
    hard_example_paths: dict[str, Path]
    reused_predictions: bool


def run_diagnostic_slicing(config: ExperimentConfig) -> DiagnosticSlicing:
    """Slice the configured experiment, training only when no predictions exist."""
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    if _has_saved_predictions(experiment_dir):
        paths = _slice_saved_experiment(config, experiment_dir)
        return DiagnosticSlicing(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            dataset=config.dataset,
            models=tuple(config.models),
            output_paths=paths,
            reused_predictions=True,
        )
    outcome = run_phase2(config)
    paths = {
        model_name: slicing_directory(outcome.output_dir)
        / f"{outcome.result.dataset}_{model_name}_slices.json"
        for model_name in outcome.result.models
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Slicing did not write {joined}.")
    return DiagnosticSlicing(
        experiment_id=outcome.experiment_id,
        experiment_dir=outcome.output_dir,
        dataset=outcome.result.dataset,
        models=outcome.result.models,
        output_paths=paths,
        reused_predictions=False,
    )


def run_diagnostic_hard_examples(config: ExperimentConfig) -> DiagnosticHardExamples:
    """Rank hard examples, training only when no predictions exist."""
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    if _has_saved_predictions(experiment_dir):
        paths = _hard_examples_from_saved(config, experiment_dir)
        return DiagnosticHardExamples(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            dataset=config.dataset,
            models=tuple(config.models),
            output_paths=paths,
            reused_predictions=True,
        )
    outcome = run_phase2(config)
    paths = _expected_hard_example_paths(outcome)
    _require_files(paths)
    return DiagnosticHardExamples(
        experiment_id=outcome.experiment_id,
        experiment_dir=outcome.output_dir,
        dataset=outcome.result.dataset,
        models=outcome.result.models,
        output_paths=paths,
        reused_predictions=False,
    )


def run_diagnostic_robustness(
    config: ExperimentConfig,
    *,
    noise_levels: Sequence[float] | None = None,
    n_seeds: int | None = None,
    features: Sequence[str] | None = None,
) -> DiagnosticRobustness:
    """Score saved models on noisy test copies, training only if no model exists."""
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    reused_model = _has_saved_models(experiment_dir, config.models)
    if not reused_model:
        outcome = run_phase2(config)
        experiment_id = outcome.experiment_id
        experiment_dir = outcome.output_dir
    paths = _robustness_from_saved(
        config,
        experiment_dir,
        noise_levels=DEFAULT_NOISE_LEVELS if noise_levels is None else noise_levels,
        n_seeds=DEFAULT_N_SEEDS if n_seeds is None else n_seeds,
        features=features,
    )
    _require_files(paths)
    return DiagnosticRobustness(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        dataset=config.dataset,
        models=tuple(config.models),
        output_paths=paths,
        reused_model=reused_model,
        comparison=compare_robustness_files(tuple(paths.values())),
    )


def run_diagnostics(config: ExperimentConfig) -> DiagnosticReport:
    """Slice and rank hard examples, training only when no predictions exist."""
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    if _has_saved_predictions(experiment_dir):
        loaded = _open_saved_experiment(config, experiment_dir)
        return DiagnosticReport(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            dataset=config.dataset,
            models=tuple(config.models),
            slice_paths=_slice_loaded(config, experiment_dir, loaded),
            hard_example_paths=_hard_examples_loaded(config, experiment_dir, loaded),
            reused_predictions=True,
        )
    outcome = run_phase2(config)
    slice_paths = {
        model_name: slicing_directory(outcome.output_dir)
        / f"{outcome.result.dataset}_{model_name}_slices.json"
        for model_name in outcome.result.models
    }
    hard_example_paths = _expected_hard_example_paths(outcome)
    _require_files(slice_paths)
    _require_files(hard_example_paths)
    return DiagnosticReport(
        experiment_id=outcome.experiment_id,
        experiment_dir=outcome.output_dir,
        dataset=outcome.result.dataset,
        models=outcome.result.models,
        slice_paths=slice_paths,
        hard_example_paths=hard_example_paths,
        reused_predictions=False,
    )


def format_diagnostic_summary(run: DiagnosticSlicing) -> str:
    """Return a short summary of the slice files that were written."""
    return _format_written(
        f"Diagnostics for {run.experiment_id}",
        run.dataset,
        run.models,
        run.reused_predictions,
        run.output_paths,
    )


def format_hard_example_summary(run: DiagnosticHardExamples) -> str:
    """Return a short summary of the hard-example files that were written."""
    return _format_written(
        f"Hard examples for {run.experiment_id}",
        run.dataset,
        run.models,
        run.reused_predictions,
        run.output_paths,
    )


def format_robustness_summary(run: DiagnosticRobustness) -> str:
    """Return a short summary of the robustness files and model comparison."""
    source = (
        "Reused the saved model."
        if run.reused_model
        else "No saved model was found, so the experiment was run first."
    )
    lines = [
        f"Robustness for {run.experiment_id}",
        f"Dataset: {run.dataset}",
        f"Models: {', '.join(run.models)}",
        source,
    ]
    lines.extend(f"{model_name}: {path}" for model_name, path in run.output_paths.items())
    if len(run.output_paths) > 1:
        lines.extend(["", run.comparison.to_string(index=False)])
    return "\n".join(lines)


def format_diagnostic_report(run: DiagnosticReport) -> str:
    """Return a short summary of the slice and hard-example files."""
    source = (
        "Reused saved predictions."
        if run.reused_predictions
        else "No saved predictions were found, so the experiment was run first."
    )
    lines = [
        f"Diagnostics for {run.experiment_id}",
        f"Dataset: {run.dataset}",
        f"Models: {', '.join(run.models)}",
        source,
        "Error slices:",
    ]
    lines.extend(f"{model_name}: {path}" for model_name, path in run.slice_paths.items())
    lines.append("Hard examples:")
    lines.extend(
        f"{model_name}: {path}" for model_name, path in run.hard_example_paths.items()
    )
    return "\n".join(lines)


def _has_saved_models(experiment_dir: Path, models: Sequence[str]) -> bool:
    if not _has_saved_predictions(experiment_dir):
        return False
    metadata = _read_json(experiment_dir / "metadata.json")
    filenames = metadata.get("filenames")
    if not isinstance(filenames, dict):
        return False
    for model_name in models:
        filename = filenames.get(model_name)
        if not isinstance(filename, str) or not (experiment_dir / filename).is_file():
            return False
    return True


def _robustness_from_saved(
    config: ExperimentConfig,
    experiment_dir: Path,
    *,
    noise_levels: Sequence[float],
    n_seeds: int,
    features: Sequence[str] | None,
) -> dict[str, Path]:
    metadata = _read_json(experiment_dir / "metadata.json")
    predictions = _read_json(experiment_dir / "predictions.json")
    _require_same_experiment(config, metadata)
    split = load_dataset(
        config.dataset,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    sample_ids = predictions.get("test_index")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("Saved predictions are missing test_index.")
    test_features = _align_features(split.X_test, sample_ids)
    _require_same_labels(split.y_test, test_features.index, predictions.get("y_test"))
    labels = split.y_test.loc[test_features.index]
    written: dict[str, Path] = {}
    destination = robustness_directory(experiment_dir)
    filenames = metadata.get("filenames")
    if not isinstance(filenames, dict):
        raise ValueError("Saved experiment metadata is missing model filenames.")
    for model_name in config.models:
        filename = filenames.get(model_name)
        if not isinstance(filename, str):
            raise ValueError(f"Saved experiment metadata has no file for {model_name!r}.")
        model_path = experiment_dir / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved model not found: {model_path}.")
        model = joblib.load(model_path)
        result = run_robustness(
            model,
            test_features,
            labels,
            dataset=config.dataset,
            model_name=model_name,
            numeric_features=metadata.get("numeric_features"),
            categorical_features=metadata.get("categorical_features"),
            features=features,
            noise_levels=noise_levels,
            n_seeds=n_seeds,
            random_state=config.random_state,
        )
        written[model_name] = save_robustness_result(result, destination)
    return written


def _has_saved_predictions(experiment_dir: Path) -> bool:
    return (experiment_dir / "predictions.json").is_file() and (
        experiment_dir / "metadata.json"
    ).is_file()


def _slice_saved_experiment(config: ExperimentConfig, experiment_dir: Path) -> dict[str, Path]:
    return _slice_loaded(config, experiment_dir, _open_saved_experiment(config, experiment_dir))


def _hard_examples_from_saved(config: ExperimentConfig, experiment_dir: Path) -> dict[str, Path]:
    return _hard_examples_loaded(
        config,
        experiment_dir,
        _open_saved_experiment(config, experiment_dir),
    )


def _open_saved_experiment(config: ExperimentConfig, experiment_dir: Path) -> tuple[dict, dict, object]:
    metadata = _read_json(experiment_dir / "metadata.json")
    predictions = _read_json(experiment_dir / "predictions.json")
    _require_same_experiment(config, metadata)
    split = load_dataset(
        config.dataset,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    return metadata, predictions, split


def _slice_loaded(
    config: ExperimentConfig,
    experiment_dir: Path,
    loaded: tuple[dict, dict, object],
) -> dict[str, Path]:
    metadata, predictions, split = loaded
    written: dict[str, Path] = {}
    destination = slicing_directory(experiment_dir)
    for model_name in config.models:
        store, features = _stored_rows(predictions, model_name, split)
        slicing = run_error_slicing(
            features,
            store.frame["y_true"].to_numpy(),
            store.frame["y_pred"].to_numpy(),
            dataset=config.dataset,
            model=model_name,
            numeric_features=metadata.get("numeric_features"),
            categorical_features=metadata.get("categorical_features"),
        )
        written[model_name] = save_slicing_result(slicing, destination)
    return written


def _hard_examples_loaded(
    config: ExperimentConfig,
    experiment_dir: Path,
    loaded: tuple[dict, dict, object],
) -> dict[str, Path]:
    _metadata, predictions, split = loaded
    written: dict[str, Path] = {}
    destination = hard_examples_directory(experiment_dir)
    for model_name in config.models:
        store, features = _stored_rows(predictions, model_name, split)
        explored = explore_hard_examples(
            store,
            features=features,
            dataset=config.dataset,
            model=model_name,
        )
        written[model_name] = save_hard_examples(explored, destination)
    return written


def _stored_rows(predictions: dict, model_name: str, split) -> tuple[PredictionStore, pd.DataFrame]:
    """Join saved predictions to the reloaded test rows by sample id."""
    store = PredictionStore.from_phase2_predictions(predictions, model_name)
    features = _align_features(split.X_test, store.frame["sample_id"].tolist())
    _require_same_labels(split.y_test, features.index, store.frame["y_true"].to_numpy())
    return store, features


def _expected_hard_example_paths(outcome) -> dict[str, Path]:
    return {
        model_name: hard_examples_directory(outcome.output_dir)
        / f"{outcome.result.dataset}_{model_name}_hard_examples.json"
        for model_name in outcome.result.models
    }


def _require_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Diagnostics did not write {joined}.")


def _format_written(title: str, dataset: str, models, reused: bool, paths: dict[str, Path]) -> str:
    source = (
        "Reused saved predictions."
        if reused
        else "No saved predictions were found, so the experiment was run first."
    )
    lines = [
        title,
        f"Dataset: {dataset}",
        f"Models: {', '.join(models)}",
        source,
    ]
    lines.extend(f"{model_name}: {path}" for model_name, path in paths.items())
    return "\n".join(lines)


def _require_same_experiment(config: ExperimentConfig, metadata: dict) -> None:
    if metadata.get("dataset") != config.dataset:
        raise ValueError(
            "Saved experiment dataset does not match the configuration: "
            f"{metadata.get('dataset')!r} and {config.dataset!r}."
        )
    if int(metadata.get("random_state")) != int(config.random_state):
        raise ValueError(
            "Saved experiment random_state does not match the configuration: "
            f"{metadata.get('random_state')} and {config.random_state}."
        )
    saved_size = float(metadata.get("test_size"))
    if abs(saved_size - float(config.test_size)) > 1e-12:
        raise ValueError(
            "Saved experiment test_size does not match the configuration: "
            f"{saved_size} and {config.test_size}."
        )


def _align_features(features: pd.DataFrame, sample_ids: list) -> pd.DataFrame:
    try:
        aligned = features.loc[list(sample_ids)]
    except KeyError as exc:
        raise ValueError(
            "Saved prediction sample ids were not found in the reloaded test split."
        ) from exc
    if len(aligned) != len(sample_ids) or _id_list(aligned.index) != _id_list(sample_ids):
        raise ValueError(
            "Saved prediction sample ids do not match the reloaded test split."
        )
    return aligned


def _require_same_labels(y_test, index, stored_labels) -> None:
    reloaded = np.asarray(y_test.loc[index].to_numpy())
    saved = np.asarray(stored_labels)
    if reloaded.shape != saved.shape or not np.array_equal(reloaded, saved):
        raise ValueError("Saved labels do not match the reloaded test split.")


def _id_list(values) -> list:
    return [_identifier(value) for value in values]


def _identifier(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return str(value)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot read {path.name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload
