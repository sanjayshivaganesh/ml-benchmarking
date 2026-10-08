"""Run one dataset against one or more selected baseline models.

Training goes through ``train_one``. Metrics and error analysis use the
single-model Phase 1 functions. This module does not call the full-grid
batch writers, and it stores fitted pipelines under ``outputs/experiments``
unless the caller passes another directory. After scoring, it writes one
prediction CSV, one slice file, and one hard-example file per selected
model from the predictions already stored on the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.analysis.error_analysis import analyze_errors
from src.data.data_loader import load_dataset
from src.diagnostics.hard_examples import save_experiment_hard_examples
from src.diagnostics.prediction_store import save_experiment_predictions
from src.diagnostics.slicing import save_experiment_slices
from src.evaluation.evaluate import evaluate_model
from src.experiments.registry import resolve_models, validate_dataset
from src.models.models import get_models
from src.training.train import TrainedModel, train_one

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = _PROJECT_ROOT / "outputs" / "experiments"


@dataclass(frozen=True, eq=False)
class ModelResult:
    """One fitted pipeline and the scores computed from it."""

    model: str
    trained: TrainedModel
    pipeline: Any
    metrics: dict[str, float]
    error_analysis: dict[str, Any]
    predictions: np.ndarray
    positive_class_probabilities: np.ndarray


@dataclass(frozen=True, eq=False)
class ExperimentResult:
    """In-memory result for one dataset and the models selected for it."""

    dataset: str
    models: tuple[str, ...]
    random_state: int
    test_size: float
    class_labels: dict[Any, Any]
    metrics: dict[str, dict[str, float]]
    error_analysis: dict[str, dict[str, Any]]
    fitted: dict[str, ModelResult]
    metadata: dict[str, Any]
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]


def run_experiment(
    dataset: str,
    models: str | list[str] | tuple[str, ...],
    random_state: int = 42,
    test_size: float = 0.2,
    model_dir: Path | str | None = None,
) -> ExperimentResult:
    """Train, score, and inspect one dataset for the selected models.

    Parameters
    ----------
    dataset:
        Dataset name understood by ``load_dataset``.
    models:
        One model name, a list of names, or ``"all"``. Names come from the
        model registry. ``"all"`` expands to every registered model.
    random_state:
        Seed forwarded to the split and the classifiers.
    test_size:
        Fraction of rows held out for evaluation. Defaults to 0.2. The same
        value is used for the scored split and for training.
    model_dir:
        Directory for fitted pipelines. Defaults to ``outputs/experiments``.
    """
    dataset_name = validate_dataset(dataset)
    selected = resolve_models(models)
    available = get_models(random_state=random_state)
    _require_known_models(selected, available)

    split = load_dataset(dataset_name, test_size=test_size, random_state=random_state)
    output_dir = Path(model_dir) if model_dir is not None else DEFAULT_EXPERIMENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fitted: dict[str, ModelResult] = {}
    for model_name in selected:
        trained = train_one(
            dataset=dataset_name,
            model=model_name,
            random_state=random_state,
            test_size=test_size,
            model_dir=output_dir,
        )
        if trained.metadata["test_size"] != split.metadata["test_size"]:
            raise RuntimeError(
                "Training and evaluation used different test_size values: "
                f"{trained.metadata['test_size']} and {split.metadata['test_size']}."
            )
        if trained.metadata["random_state"] != random_state:
            raise RuntimeError(
                "Training and evaluation used different random_state values: "
                f"{trained.metadata['random_state']} and {random_state}."
            )
        pipeline = joblib.load(trained.path)
        model_result = _score_model(pipeline, trained, split)
        fitted[model_name] = model_result

    class_labels = dict(split.metadata["class_labels"])
    result = ExperimentResult(
        dataset=dataset_name,
        models=selected,
        random_state=random_state,
        test_size=float(split.metadata["test_size"]),
        class_labels=class_labels,
        metrics={name: fitted[name].metrics for name in selected},
        error_analysis={name: fitted[name].error_analysis for name in selected},
        fitted=fitted,
        metadata=_experiment_metadata(
            split=split,
            selected=selected,
            fitted=fitted,
            random_state=random_state,
            model_dir=output_dir,
            class_labels=class_labels,
        ),
        X_test=split.X_test.copy(),
        y_test=split.y_test.copy(),
        feature_names=list(split.feature_names),
    )
    save_experiment_predictions(result, output_dir)
    save_experiment_slices(result, output_dir)
    save_experiment_hard_examples(result, output_dir)
    return result


def _score_model(pipeline, trained: TrainedModel, split) -> ModelResult:
    metrics = evaluate_model(pipeline, split.X_test, split.y_test)
    error_analysis = analyze_errors(
        pipeline,
        split.X_test,
        split.y_test,
        feature_names=split.feature_names,
    )
    predictions = np.asarray(pipeline.predict(split.X_test))
    positive_class_probabilities = _positive_class_probabilities(pipeline, split.X_test)
    return ModelResult(
        model=trained.model,
        trained=trained,
        pipeline=pipeline,
        metrics=metrics,
        error_analysis=error_analysis,
        predictions=predictions,
        positive_class_probabilities=positive_class_probabilities,
    )


def _positive_class_probabilities(pipeline, features) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(features), dtype=float)
    classes = np.asarray(getattr(pipeline, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(
            "Expected binary classes that include positive label 1, "
            f"found {classes.tolist()}."
        )
    positive = probabilities[:, int(matches[0])]
    if not np.isfinite(positive).all() or np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError(
            "Positive-class probabilities must be finite and inside [0, 1]."
        )
    return positive


def _experiment_metadata(
    split,
    selected: tuple[str, ...],
    fitted: dict[str, ModelResult],
    random_state: int,
    model_dir: Path,
    class_labels: dict[Any, Any],
) -> dict[str, Any]:
    return {
        "dataset": split.metadata["name"],
        "models": list(selected),
        "random_state": random_state,
        "test_size": split.metadata["test_size"],
        "stratified": bool(split.metadata["stratified"]),
        "source": split.metadata["source"],
        "class_labels": dict(class_labels),
        "feature_names": list(split.feature_names),
        "numeric_features": list(split.metadata["numeric_features"]),
        "categorical_features": list(split.metadata["categorical_features"]),
        "n_train_samples": int(len(split.y_train)),
        "n_test_samples": int(len(split.y_test)),
        "model_dir": str(model_dir),
        "filenames": {
            name: fitted[name].trained.path.name for name in selected
        },
        "hyperparameters": {
            name: dict(fitted[name].trained.metadata["hyperparameters"])
            for name in selected
        },
    }


def _require_known_models(selected: tuple[str, ...], available: dict[str, Any]) -> None:
    unknown = [name for name in selected if name not in available]
    if unknown:
        supported = ", ".join(available)
        joined = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Unknown model(s) {joined}. Available models: {supported}.")
