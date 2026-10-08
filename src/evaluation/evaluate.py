"""Evaluate trained classifiers on held-out test data.

Metrics that use class labels come from ``predict``. ROC-AUC and Brier score
use positive-class probabilities from ``predict_proba``. Evaluation never
fits a model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.data_loader import SUPPORTED_DATASETS, load_dataset
from src.models.models import MODEL_NAMES
from src.training.train import DEFAULT_MODEL_DIR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = _PROJECT_ROOT / "outputs" / "metrics" / "metrics.json"

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "brier_score",
)


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Score one fitted classifier on a held-out test set.

    Accuracy, precision, recall, and F1 use ``predict`` at the estimator's
    default 0.5 threshold. ROC-AUC and Brier score use the probability of
    class ``1``. Models without ``predict_proba`` raise instead of returning
    stand-in scores.
    """
    y_true = _as_label_array(y_test)
    predictions = np.asarray(model.predict(X_test))
    if predictions.shape[0] != y_true.shape[0]:
        raise ValueError(
            "Predictions and y_test have different lengths: "
            f"{predictions.shape[0]} and {y_true.shape[0]}."
        )
    positive_probabilities = _positive_class_probabilities(model, X_test)
    if positive_probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(
            "Probabilities and y_test have different lengths: "
            f"{positive_probabilities.shape[0]} and {y_true.shape[0]}."
        )

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            pos_label=1,
            average="binary",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, positive_probabilities),
        "brier_score": brier_score_loss(
            y_true,
            positive_probabilities,
            pos_label=1,
        ),
    }
    return {name: _json_float(metrics[name]) for name in METRIC_NAMES}


def evaluate_all(
    model_dir: Path | str | None = None,
    metrics_path: Path | str | None = None,
    random_state: int = 42,
    datasets: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate every saved baseline model and write ``metrics.json``.

    Each model is loaded from disk and scored on the test split that matches
    its training seed. Nothing is refit.
    """
    selected = _resolve_datasets(datasets)
    artifact_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    output_path = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH

    results: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_name in selected:
        results[dataset_name] = {}
        for model_name in MODEL_NAMES:
            path = artifact_dir / f"{dataset_name}__{model_name}.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing trained model for {dataset_name}/{model_name}: {path}. "
                    "Train the benchmark models before evaluation."
                )
            model = joblib.load(path)
            split = _held_out_split(dataset_name, model, random_state)
            results[dataset_name][model_name] = evaluate_model(
                model,
                split.X_test,
                split.y_test,
            )

    _write_metrics(results, output_path)
    return results


def _positive_class_probabilities(model, features) -> np.ndarray:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError(
            f"{type(model).__name__} does not implement predict_proba(). "
            "ROC-AUC and Brier score require positive-class probabilities."
        )
    probabilities = np.asarray(predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "predict_proba() must return one column per class; "
            f"got shape {tuple(probabilities.shape)}."
        )
    positive_index = _positive_class_index(model, probabilities.shape[1])
    positive = probabilities[:, positive_index]
    if not np.isfinite(positive).all() or np.any((positive < 0.0) | (positive > 1.0)):
        raise ValueError(
            "Positive-class probabilities must be finite and inside [0, 1]."
        )
    return positive


def _positive_class_index(model, n_columns: int) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        if n_columns != 2:
            raise ValueError(
                "Cannot choose the positive class without classes_ when "
                f"predict_proba() has {n_columns} columns."
            )
        return 1
    class_labels = np.asarray(classes)
    matches = np.flatnonzero(class_labels == 1)
    if len(matches) != 1:
        raise ValueError(
            "Expected binary classes that include positive label 1, "
            f"found {class_labels.tolist()}."
        )
    index = int(matches[0])
    if index >= n_columns:
        raise ValueError(
            f"Positive class index {index} is outside predict_proba() width {n_columns}."
        )
    return index


def _as_label_array(y_test) -> np.ndarray:
    labels = np.asarray(y_test)
    if labels.ndim > 1:
        labels = np.ravel(labels)
    return labels


def _held_out_split(dataset_name: str, model, random_state: int):
    metadata = getattr(model, "metadata", None) or {}
    return load_dataset(
        dataset_name,
        test_size=metadata.get("test_size", 0.2),
        random_state=metadata.get("random_state", random_state),
    )


def _resolve_datasets(
    datasets: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if datasets is None:
        return SUPPORTED_DATASETS
    selected = tuple(datasets)
    if not selected:
        raise ValueError("datasets must contain at least one dataset name.")
    unknown = [name for name in selected if name not in SUPPORTED_DATASETS]
    if unknown:
        supported = ", ".join(SUPPORTED_DATASETS)
        joined = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Unknown dataset(s) {joined}. Supported names: {supported}.")
    return selected


def _json_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric value is not a finite float: {value!r}.")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Metric value {number} is outside [0, 1].")
    return number


def _write_metrics(results: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
