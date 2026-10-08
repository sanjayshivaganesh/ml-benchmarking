"""Slice existing predictions by raw test-set features.

Each slice reports how performance differs inside a subgroup. A difference
is an association with that subgroup. It is not evidence that the feature
caused the errors. The model is not retrained, and predictions are not
recomputed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.evaluate import LABEL_METRIC_NAMES, score_predictions

MIN_SLICE_SIZE = 20
DEFAULT_NUMERIC_BINS = 4
_ASSOCIATION_NOTE = (
    "Slice metrics describe performance differences within subgroups of the "
    "test set. They do not show that a feature caused the model to fail. "
    "An F1 delta is omitted when the slice has no class 1 rows, because "
    "binary F1 is then not a measure of error."
)


@dataclass(frozen=True)
class ErrorSlicingResult:
    """Slice metrics for one dataset and model, plus a table for later reports."""

    dataset: str | None
    model: str | None
    overall_metrics: dict[str, float]
    min_slice_size: int
    numeric_bins: int
    slices: tuple[dict[str, Any], ...]
    table: pd.DataFrame
    worst_slices: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description of the slices."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "min_slice_size": self.min_slice_size,
            "numeric_bins": self.numeric_bins,
            "association_only": True,
            "note": _ASSOCIATION_NOTE,
            "overall_metrics": dict(self.overall_metrics),
            "worst_slices": [_json_slice(item) for item in self.worst_slices],
            "slices": [_json_slice(item) for item in self.slices],
        }


def run_error_slicing(
    X_test,
    y_test,
    predictions,
    *,
    dataset: str | None = None,
    model: str | None = None,
    numeric_features: list[str] | tuple[str, ...] | None = None,
    categorical_features: list[str] | tuple[str, ...] | None = None,
    n_bins: int = DEFAULT_NUMERIC_BINS,
    min_slice_size: int = MIN_SLICE_SIZE,
) -> ErrorSlicingResult:
    """Compare subgroup performance using predictions that already exist.

    Numerical features use quantile bins, labeled ``Q1`` through ``Qn`` from
    low to high. Duplicate bin edges are dropped. A constant feature becomes
    one slice instead of invalid bins. Categorical features group by category.
    Missing values are their own slice, ``(missing)``.

    A slice with fewer than ``min_slice_size`` rows is kept and marked
    unreliable. It is not included in ``worst_slices``. A large slice with no
    class 1 rows also stays out of that ranking, because its binary F1 is 0
    even when every prediction is correct. Worst slices are the remaining
    slices with the largest F1 drop relative to the full test set.
    """
    features = _feature_frame(X_test)
    labels = _aligned_labels(y_test, features.shape[0], "y_test")
    predicted = _aligned_labels(predictions, features.shape[0], "predictions")
    bin_count = _require_bin_count(n_bins)
    minimum = _require_min_slice_size(min_slice_size)
    numeric_names = _name_set(numeric_features)
    categorical_names = _name_set(categorical_features)
    overall = score_predictions(labels, predicted)

    slices: list[dict[str, Any]] = []
    for column in features.columns:
        kind = _feature_kind(features[column], column, numeric_names, categorical_names)
        if kind == "numeric":
            grouped = _numeric_groups(features[column], bin_count)
        else:
            grouped = _categorical_groups(features[column])
        for definition, positions in grouped:
            count = int(len(positions))
            if count == 0:
                continue
            metrics = score_predictions(labels[positions], predicted[positions])
            reliable = count >= minimum
            f1_comparable = reliable and _has_positive_class(labels[positions])
            slices.append(
                {
                    "feature": str(column),
                    "feature_type": kind,
                    "slice_definition": definition,
                    "sample_count": count,
                    "reliable": reliable,
                    "min_slice_size": minimum,
                    "f1_comparable": f1_comparable,
                    "metrics": metrics,
                    "delta_accuracy": (
                        metrics["accuracy"] - overall["accuracy"] if reliable else None
                    ),
                    "delta_f1": metrics["f1"] - overall["f1"] if f1_comparable else None,
                }
            )

    ranked = _rank_worst(slices)
    for index, item in enumerate(ranked, start=1):
        item["worst_rank"] = index
    table = _slice_table(slices)
    return ErrorSlicingResult(
        dataset=dataset,
        model=model,
        overall_metrics=overall,
        min_slice_size=minimum,
        numeric_bins=bin_count,
        slices=tuple(slices),
        table=table,
        worst_slices=tuple(ranked),
    )


def save_slicing_result(result: ErrorSlicingResult, directory: Path | str) -> Path:
    """Write ``<dataset>_<model>_slices.json`` under ``directory``."""
    if not result.dataset or not result.model:
        raise ValueError("Saving slices requires both a dataset name and a model name.")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.dataset}_{result.model}_slices.json"
    payload = result.to_dict()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def slicing_directory(model_dir: Path | str) -> Path:
    """Return ``outputs/diagnostics/slicing`` for a normal experiment directory.

    A temporary model directory keeps its JSON inside that directory so tests
    do not write into the project tree.
    """
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "slicing"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "slicing"
    return directory / "diagnostics" / "slicing"


def save_experiment_slices(result, model_dir: Path | str) -> dict[str, Path]:
    """Slice each selected model from the predictions stored on ``result``."""
    written: dict[str, Path] = {}
    destination = slicing_directory(model_dir)
    metadata = getattr(result, "metadata", {}) or {}
    numeric = metadata.get("numeric_features")
    categorical = metadata.get("categorical_features")
    for model_name in result.models:
        fitted = result.fitted[model_name]
        slicing = run_error_slicing(
            result.X_test,
            result.y_test,
            fitted.predictions,
            dataset=result.dataset,
            model=model_name,
            numeric_features=numeric,
            categorical_features=categorical,
        )
        written[model_name] = save_slicing_result(slicing, destination)
    return written


def _feature_frame(features) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("X_test must be a DataFrame of raw feature rows.")
    if features.shape[0] == 0:
        raise ValueError("X_test must contain at least one sample.")
    frame = features.copy()
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("X_test column names must be unique.")
    return frame


def _aligned_labels(values, n_rows: int, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim > 1:
        labels = np.ravel(labels)
    if labels.shape[0] != n_rows:
        raise ValueError(
            f"{name} and X_test have different lengths: {labels.shape[0]} and {n_rows}."
        )
    return labels


def _require_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer.")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2.")
    return int(n_bins)


def _require_min_slice_size(min_slice_size: int) -> int:
    if isinstance(min_slice_size, bool) or not isinstance(min_slice_size, (int, np.integer)):
        raise TypeError("min_slice_size must be an integer.")
    if int(min_slice_size) < 1:
        raise ValueError("min_slice_size must be at least 1.")
    return int(min_slice_size)


def _name_set(names) -> set[str] | None:
    if names is None:
        return None
    return {str(name) for name in names}


def _feature_kind(
    series: pd.Series,
    name: str,
    numeric_names: set[str] | None,
    categorical_names: set[str] | None,
) -> str:
    if numeric_names is not None and name in numeric_names:
        return "numeric"
    if categorical_names is not None and name in categorical_names:
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _numeric_groups(series: pd.Series, n_bins: int) -> list[tuple[str, np.ndarray]]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(numeric)
    known_positions = np.flatnonzero(~missing)
    groups: list[tuple[str, np.ndarray]] = []
    if known_positions.size:
        known_values = numeric[known_positions]
        nunique = int(pd.Series(known_values).nunique(dropna=True))
        if nunique < 2:
            definition = f"constant = {_format_number(known_values[0])}"
            groups.append((definition, known_positions))
        else:
            binned = pd.Series(
                pd.qcut(
                    known_values,
                    q=min(n_bins, nunique),
                    duplicates="drop",
                )
            )
            intervals = list(binned.cat.categories)
            codes = binned.cat.codes.to_numpy()
            for index, interval in enumerate(intervals, start=1):
                positions = known_positions[codes == index - 1]
                groups.append((f"Q{index}: {interval}", positions))
    if missing.any():
        groups.append(("(missing)", np.flatnonzero(missing)))
    return groups


def _categorical_groups(series: pd.Series) -> list[tuple[str, np.ndarray]]:
    labels = [_category_label(value) for value in series.to_numpy()]
    grouped: dict[str, list[int]] = {}
    for position, label in enumerate(labels):
        grouped.setdefault(label, []).append(position)
    return [
        (label, np.asarray(positions, dtype=int))
        for label, positions in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _category_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "(missing)"
    return str(value)


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".6g")


def _has_positive_class(labels: np.ndarray) -> bool:
    """Return whether binary F1 for class 1 is defined on these labels."""
    for value in np.asarray(labels).tolist():
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, np.integer)) and int(value) == 1:
            return True
        if isinstance(value, (float, np.floating)) and float(value) == 1.0:
            return True
    return False


def _rank_worst(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reliable = [
        item for item in slices if item["reliable"] and item["delta_f1"] is not None
    ]
    return sorted(
        reliable,
        key=lambda item: (
            item["delta_f1"],
            item["delta_accuracy"],
            item["feature"],
            item["slice_definition"],
        ),
    )


def _slice_table(slices: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "feature",
        "feature_type",
        "slice_definition",
        "sample_count",
        "reliable",
        "min_slice_size",
        *LABEL_METRIC_NAMES,
        "delta_accuracy",
        "delta_f1",
        "worst_rank",
    ]
    rows = []
    for item in slices:
        row = {
            "feature": item["feature"],
            "feature_type": item["feature_type"],
            "slice_definition": item["slice_definition"],
            "sample_count": item["sample_count"],
            "reliable": item["reliable"],
            "min_slice_size": item["min_slice_size"],
            "delta_accuracy": item["delta_accuracy"],
            "delta_f1": item["delta_f1"],
            "worst_rank": item.get("worst_rank"),
        }
        row.update(item["metrics"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _json_slice(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "feature": item["feature"],
        "feature_type": item["feature_type"],
        "slice_definition": item["slice_definition"],
        "sample_count": item["sample_count"],
        "reliable": item["reliable"],
        "min_slice_size": item["min_slice_size"],
        "f1_comparable": item["f1_comparable"],
        "metrics": dict(item["metrics"]),
        "delta_accuracy": item["delta_accuracy"],
        "delta_f1": item["delta_f1"],
    }
    if "worst_rank" in item:
        payload["worst_rank"] = item["worst_rank"]
    return payload
