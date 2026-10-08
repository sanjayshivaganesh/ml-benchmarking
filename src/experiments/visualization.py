"""Plot and tabulate one experiment result.

Inputs are the metrics, labels, probabilities, and confusion matrices already
stored on an ``ExperimentResult``. This module does not load data, train, or
recompute those metrics.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from src.evaluation.evaluate import METRIC_NAMES

_ROC_DISPLAY_DECIMALS = 4


def metric_comparison_table(result) -> pd.DataFrame:
    """Return one row per selected model using the stored evaluation metrics.

    Columns follow the Phase 1 metric names that appear for at least one
    model. A metric missing for a particular model is left empty.
    """
    models = _selected_models(result)
    stored = _metric_map(result)
    available = [
        name
        for name in METRIC_NAMES
        if any(name in stored[model] for model in models)
    ]
    if not available:
        raise ValueError("No evaluation metrics are available to tabulate.")
    rows = []
    for model in models:
        row: dict[str, Any] = {"model": model}
        for name in available:
            value = stored[model].get(name)
            row[name] = np.nan if value is None else float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", *available])


def plot_roc_curves(result, ax: Axes | None = None) -> Axes:
    """Draw one ROC curve per selected model on a single axes.

    Curve points come from the stored true labels and class-1 probabilities.
    The ROC-AUC shown in the legend is the stored ``roc_auc`` metric.
    """
    models = _selected_models(result)
    labels = _true_labels(result)
    stored = _metric_map(result)
    created = ax is None
    if ax is None:
        _, ax = plt.subplots()
    try:
        for model in models:
            probabilities = _probabilities(result, model)
            if probabilities.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{model} probabilities and true labels have different lengths: "
                    f"{probabilities.shape[0]} and {labels.shape[0]}."
                )
            if "roc_auc" not in stored[model]:
                raise ValueError(f"ROC-AUC for {model!r} is missing.")
            try:
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    labels,
                    probabilities,
                    pos_label=1,
                )
            except ValueError as exc:
                raise ValueError(f"Cannot draw an ROC curve for {model!r}. {exc}") from exc
            auc_value = float(stored[model]["roc_auc"])
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model} (ROC-AUC {auc_value:.{_ROC_DISPLAY_DECIMALS}f})",
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.6", label="chance")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        dataset = getattr(result, "dataset", None)
        ax.set_title(f"ROC curve: {dataset}" if dataset else "ROC curve")
        ax.legend(loc="lower right")
    except Exception:
        if created:
            plt.close(ax.figure)
        raise
    return ax


def plot_confusion_matrix(result, model: str, ax: Axes | None = None) -> Axes:
    """Draw one model's stored confusion matrix.

    The matrix is ``[[TN, FP], [FN, TP]]``. Tick labels use the experiment's
    class names.
    """
    matrix = _confusion_matrix(result, model)
    tick_labels = _tick_labels(result)
    created = ax is None
    if ax is None:
        _, ax = plt.subplots()
    try:
        image = ax.imshow(matrix, cmap="Blues")
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(str(model))
        threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
        for row in range(2):
            for column in range(2):
                count = int(matrix[row, column])
                color = "white" if matrix[row, column] > threshold else "black"
                ax.text(column, row, str(count), ha="center", va="center", color=color)
    except Exception:
        if created:
            plt.close(ax.figure)
        raise
    return ax


def plot_confusion_matrices(result) -> Figure:
    """Draw the stored confusion matrix for every selected model."""
    models = _selected_models(result)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.5 * len(models), 4.2),
        squeeze=False,
    )
    try:
        for ax, model in zip(axes[0], models):
            plot_confusion_matrix(result, model, ax=ax)
        fig.tight_layout()
    except Exception:
        plt.close(fig)
        raise
    return fig


def _selected_models(result) -> tuple[str, ...]:
    models = getattr(result, "models", None)
    if models is None:
        raise ValueError("The experiment result has no selected models.")
    selected = tuple(models)
    if not selected:
        raise ValueError("No models are available to visualize.")
    return selected


def _metric_map(result) -> dict[str, dict[str, float]]:
    stored = getattr(result, "metrics", None)
    if not isinstance(stored, dict):
        raise ValueError("Evaluation metrics are missing from the experiment result.")
    models = _selected_models(result)
    resolved: dict[str, dict[str, float]] = {}
    for model in models:
        scores = stored.get(model)
        if not isinstance(scores, dict):
            available = ", ".join(str(name) for name in stored)
            raise ValueError(
                f"Metrics for {model!r} are missing. Available models: {available}."
            )
        resolved[model] = scores
    return resolved


def _true_labels(result) -> np.ndarray:
    labels = getattr(result, "y_test", None)
    if labels is None:
        raise ValueError("True labels are missing from the experiment result.")
    values = np.asarray(labels)
    if values.ndim > 1:
        values = np.ravel(values)
    if values.size == 0:
        raise ValueError("True labels are empty.")
    return values


def _probabilities(result, model: str) -> np.ndarray:
    fitted = getattr(result, "fitted", None)
    if not isinstance(fitted, dict) or model not in fitted:
        raise ValueError(f"Class-1 probabilities for {model!r} are missing.")
    model_result = fitted[model]
    probabilities = getattr(model_result, "positive_class_probabilities", None)
    if probabilities is None:
        raise ValueError(f"Class-1 probabilities for {model!r} are missing.")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        values = np.ravel(values)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"Class-1 probabilities for {model!r} must be non-empty and finite."
        )
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            f"Class-1 probabilities for {model!r} must be inside [0, 1]."
        )
    return values


def _confusion_matrix(result, model: str) -> np.ndarray:
    reports = getattr(result, "error_analysis", None)
    if not isinstance(reports, dict) or model not in reports:
        available = ", ".join(str(name) for name in _selected_models(result))
        raise ValueError(
            f"Confusion matrix for {model!r} is missing. Available models: {available}."
        )
    report = reports[model]
    if not isinstance(report, dict) or "confusion_matrix" not in report:
        raise ValueError(f"Confusion matrix for {model!r} is missing.")
    labels = report.get("confusion_matrix_labels", [0, 1])
    if [int(label) for label in labels] != [0, 1]:
        raise ValueError(
            f"Confusion matrix for {model!r} must use labels [0, 1], found {labels}."
        )
    matrix = np.asarray(report["confusion_matrix"], dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(
            f"Confusion matrix for {model!r} must have shape (2, 2), found {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Confusion matrix for {model!r} contains non-finite values.")
    return matrix


def _tick_labels(result) -> list[str]:
    class_labels = getattr(result, "class_labels", None)
    if not isinstance(class_labels, dict) or not class_labels:
        raise ValueError("Class labels are missing from the experiment result.")
    return [f"{label} = {_class_name(class_labels, label)}" for label in (0, 1)]


def _class_name(class_labels: dict[Any, Any], label: int) -> str:
    for key, name in class_labels.items():
        try:
            key_label = int(key)
        except (TypeError, ValueError):
            continue
        if key_label == label:
            return str(name)
    available = ", ".join(f"{key}={name}" for key, name in class_labels.items())
    raise ValueError(
        f"Class label {label} is missing. Available class labels: {available}."
    )
