"""Find confident mistakes and uncertain predictions.

This module reads a ``PredictionStore``. It does not train or call ``predict``.
For the binary rows produced by this project, ``probability`` is the
positive-class probability ``p``. Uncertainty is ``1 - 2 * abs(p - 0.5)``:
it is 1 at ``p = 0.5`` and 0 at ``p = 0`` or ``p = 1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.diagnostics.prediction_store import PREDICTION_COLUMNS, PredictionStore

DEFAULT_TOP_N = 20
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
_CATEGORIES = (
    "correct_confident",
    "correct_uncertain",
    "wrong_confident",
    "wrong_uncertain",
)


@dataclass(frozen=True)
class HardExampleResult:
    """Ranked hard examples and counts of the four confidence/error groups."""

    dataset: str | None
    model: str | None
    top_n: int
    confidence_threshold: float
    high_confidence_errors: tuple[dict[str, Any], ...]
    most_uncertain: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    table: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description of the hard examples."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "top_n": self.top_n,
            "confidence_threshold": self.confidence_threshold,
            "summary": dict(self.summary),
            "high_confidence_errors": list(self.high_confidence_errors),
            "most_uncertain": list(self.most_uncertain),
        }


def explore_hard_examples(
    store: PredictionStore,
    *,
    features: pd.DataFrame | None = None,
    dataset: str | None = None,
    model: str | None = None,
    top_n: int = DEFAULT_TOP_N,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    class_probabilities=None,
) -> HardExampleResult:
    """Rank stored predictions by confident errors and by uncertainty.

    ``high_confidence_errors`` keeps incorrect rows, highest confidence
    first. ``most_uncertain`` keeps every row, highest uncertainty first.
    Both lists stop at ``top_n``. Each example includes the original feature
    values when the store or ``features`` has them.

    Binary uncertainty uses the positive-class probability. When
    ``class_probabilities`` is a matrix with more than two classes, uncertainty
    is the normalized entropy of that distribution. Otherwise a non-binary
    store uses ``1 - confidence``.
    """
    if not isinstance(store, PredictionStore):
        raise TypeError("explore_hard_examples expects a PredictionStore.")
    limit = _require_top_n(top_n)
    threshold = _require_threshold(confidence_threshold)
    frame = store.frame.copy()
    frame["uncertainty"] = _uncertainty(frame, class_probabilities)
    frame["category"] = [
        _category(bool(correct), float(confidence), threshold)
        for correct, confidence in zip(frame["correct"], frame["confidence"])
    ]
    feature_rows = _feature_rows(store, features)
    errors = frame.loc[~frame["correct"].to_numpy()].sort_values(
        ["confidence", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    uncertain = frame.sort_values(
        ["uncertainty", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    high_confidence_errors = tuple(
        _example(frame, feature_rows, int(position))
        for position in errors.head(limit).index
    )
    most_uncertain = tuple(
        _example(frame, feature_rows, int(position))
        for position in uncertain.head(limit).index
    )
    summary = _summary(frame, limit, threshold)
    table = frame.reset_index(drop=True)
    return HardExampleResult(
        dataset=dataset,
        model=model,
        top_n=limit,
        confidence_threshold=threshold,
        high_confidence_errors=high_confidence_errors,
        most_uncertain=most_uncertain,
        summary=summary,
        table=table,
    )


def save_hard_examples(result: HardExampleResult, directory: Path | str) -> Path:
    """Write ``<dataset>_<model>_hard_examples.json`` under ``directory``."""
    if not result.dataset or not result.model:
        raise ValueError("Saving hard examples requires both a dataset name and a model name.")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.dataset}_{result.model}_hard_examples.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def hard_examples_directory(model_dir: Path | str) -> Path:
    """Return ``outputs/diagnostics/hard_examples`` for a normal experiment directory."""
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "hard_examples"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "hard_examples"
    return directory / "diagnostics" / "hard_examples"


def save_experiment_hard_examples(result, model_dir: Path | str) -> dict[str, Path]:
    """Write hard-example files from the predictions already stored on ``result``."""
    written: dict[str, Path] = {}
    destination = hard_examples_directory(model_dir)
    for model_name in result.models:
        store = PredictionStore.from_experiment(result, model_name)
        explored = explore_hard_examples(
            store,
            dataset=result.dataset,
            model=model_name,
        )
        written[model_name] = save_hard_examples(explored, destination)
    return written


def _uncertainty(frame: pd.DataFrame, class_probabilities) -> np.ndarray:
    if class_probabilities is not None:
        matrix = np.asarray(class_probabilities, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(frame):
            raise ValueError(
                "class_probabilities must have one row per prediction, "
                f"found shape {getattr(matrix, 'shape', None)}."
            )
        if matrix.shape[1] > 2:
            return _normalized_entropy(matrix)
    probability = frame["probability"].to_numpy(dtype=float)
    if _is_binary_frame(frame):
        return 1.0 - 2.0 * np.abs(probability - 0.5)
    return 1.0 - frame["confidence"].to_numpy(dtype=float)


def _normalized_entropy(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Class probabilities must be finite and inside [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Each class-probability row must sum to 1.")
    clipped = np.clip(matrix, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / np.log(matrix.shape[1])


def _is_binary_frame(frame: pd.DataFrame) -> bool:
    labels = set(frame["y_true"].tolist()) | set(frame["y_pred"].tolist())
    normalized = set()
    for label in labels:
        if isinstance(label, (bool, np.bool_)):
            return False
        if isinstance(label, (int, np.integer)):
            normalized.add(int(label))
            continue
        if isinstance(label, (float, np.floating)) and float(label).is_integer():
            normalized.add(int(label))
            continue
        return False
    return normalized <= {0, 1}


def _category(correct: bool, confidence: float, threshold: float) -> str:
    if correct and confidence >= threshold:
        return "correct_confident"
    if correct:
        return "correct_uncertain"
    if confidence >= threshold:
        return "wrong_confident"
    return "wrong_uncertain"


def _summary(frame: pd.DataFrame, top_n: int, threshold: float) -> dict[str, Any]:
    counts = {name: 0 for name in _CATEGORIES}
    for name, count in frame["category"].value_counts().items():
        counts[str(name)] = int(count)
    counts["n_samples"] = int(len(frame))
    counts["top_n"] = top_n
    counts["confidence_threshold"] = threshold
    counts["n_errors"] = int((~frame["correct"].to_numpy()).sum())
    mean_uncertainty = _mean_uncertainty(frame["uncertainty"].to_numpy(dtype=float))
    if mean_uncertainty is not None:
        counts["mean_uncertainty"] = mean_uncertainty
    return counts


def _mean_uncertainty(values: np.ndarray) -> float | None:
    if values.size == 0 or not np.isfinite(values).all():
        return None
    if np.all(values == values[0]):
        return float(values[0])
    return float(np.mean(values))


def _feature_rows(store: PredictionStore, features: pd.DataFrame | None) -> pd.DataFrame | None:
    if features is not None:
        joined = store.join_features(features)
        columns = [column for column in joined.columns if column not in PREDICTION_COLUMNS]
        return joined.loc[:, columns].reset_index(drop=True)
    if store.features is None:
        return None
    return store.features.reset_index(drop=True)


def _example(frame: pd.DataFrame, features: pd.DataFrame | None, position: int) -> dict[str, Any]:
    row = frame.loc[position]
    if features is None:
        feature_values = None
    else:
        feature_values = {
            str(column): _json_value(features.loc[position, column]) for column in features.columns
        }
    return {
        "sample_id": _json_value(row["sample_id"]),
        "y_true": _json_value(row["y_true"]),
        "y_pred": _json_value(row["y_pred"]),
        "probability": float(row["probability"]),
        "confidence": float(row["confidence"]),
        "uncertainty": float(row["uncertainty"]),
        "correct": bool(row["correct"]),
        "category": str(row["category"]),
        "features": feature_values,
    }


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _require_top_n(top_n: int) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an integer.")
    if int(top_n) < 0:
        raise ValueError("top_n must be at least 0.")
    return int(top_n)


def _require_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
        raise TypeError("confidence_threshold must be a number.")
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("confidence_threshold must be inside (0, 1].")
    return value
