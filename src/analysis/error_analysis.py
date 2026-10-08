"""Inspect misclassified test samples and simple feature slices.

Slicing uses the original test columns passed to ``analyze_errors``. Saved
pipelines may one-hot encode features internally; those transformed columns
are not written into the error report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.data_loader import SUPPORTED_DATASETS, load_dataset
from src.models.models import MODEL_NAMES
from src.training.train import DEFAULT_MODEL_DIR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERROR_REPORT_PATH = _PROJECT_ROOT / "outputs" / "errors" / "error_report.json"
_NUMERIC_BINS = 4
_MAX_REPORTED_CATEGORIES = 20


def analyze_errors(
    model,
    X_test,
    y_test,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Describe one model's mistakes on a held-out test set.

    Predictions come from ``predict`` at the estimator's default threshold.
    Positive-class probabilities are recorded when ``predict_proba`` exists.
    The model is not refit.
    """
    y_true = _as_binary_labels(y_test, "y_test")
    predictions = _as_binary_labels(model.predict(X_test), "predictions")
    if predictions.shape[0] != y_true.shape[0]:
        raise ValueError(
            "Predictions and y_test have different lengths: "
            f"{predictions.shape[0]} and {y_true.shape[0]}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("y_test must contain at least one sample.")

    features = _feature_frame(X_test, feature_names)
    if len(features) != y_true.shape[0]:
        raise ValueError(
            "X_test and y_test have different lengths: "
            f"{len(features)} and {y_true.shape[0]}."
        )

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    misclassified_count = false_positives + false_negatives
    incorrect = predictions != y_true
    predicted_probability, positive_probability = _probabilities_for_rows(
        model,
        X_test,
        predictions,
    )
    samples = _misclassified_samples(
        features,
        y_true,
        predictions,
        incorrect,
        predicted_probability,
        positive_probability,
    )
    error_flags = incorrect.astype(int)
    feature_slices = {
        column: _slice_feature(features[column], error_flags)
        for column in _select_slice_features(features, error_flags)
    }
    return {
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": [[int(value) for value in row] for row in matrix],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "misclassified_count": misclassified_count,
        "misclassification_rate": _unit_float(misclassified_count / y_true.shape[0]),
        "n_test": int(y_true.shape[0]),
        "misclassified_samples": samples,
        "feature_slices": feature_slices,
    }


def analyze_all(
    model_dir: Path | str | None = None,
    report_path: Path | str | None = None,
    random_state: int = 42,
    datasets: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Analyze every saved baseline model and write ``error_report.json``.

    Each model is loaded and scored on the test split from its training
    metadata. Nothing is refit.
    """
    selected = _resolve_datasets(datasets)
    artifact_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    output_path = Path(report_path) if report_path is not None else DEFAULT_ERROR_REPORT_PATH

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name in selected:
        results[dataset_name] = {}
        for model_name in MODEL_NAMES:
            path = artifact_dir / f"{dataset_name}__{model_name}.joblib"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing trained model for {dataset_name}/{model_name}: {path}. "
                    "Train the benchmark models before error analysis."
                )
            model = joblib.load(path)
            split = _held_out_split(dataset_name, model, random_state)
            results[dataset_name][model_name] = analyze_errors(
                model,
                split.X_test,
                split.y_test,
                feature_names=split.feature_names,
            )

    _write_report(results, output_path)
    return results


def _misclassified_samples(
    features: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
    incorrect: np.ndarray,
    predicted_probability: list[float | None],
    positive_probability: list[float | None],
) -> list[dict[str, Any]]:
    integer_columns = {
        column: bool(pd.api.types.is_integer_dtype(features[column]))
        for column in features.columns
    }
    samples = []
    for position in np.flatnonzero(incorrect):
        true_label = int(y_true[position])
        predicted_label = int(predictions[position])
        if true_label == 0 and predicted_label == 1:
            error_type = "false_positive"
        else:
            error_type = "false_negative"
        samples.append(
            {
                "sample_index": _json_index(features.index[position]),
                "test_position": int(position),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "error_type": error_type,
                "predicted_probability": predicted_probability[position],
                "positive_class_probability": positive_probability[position],
                "features": _feature_payload(features, int(position), integer_columns),
            }
        )
    samples.sort(key=_sample_sort_key)
    return samples


def _feature_payload(
    features: pd.DataFrame,
    position: int,
    integer_columns: dict[str, bool],
) -> dict[str, Any]:
    row = features.iloc[position]
    return {
        str(column): _json_feature_value(row[column], integer_columns[column])
        for column in features.columns
    }


def _select_slice_features(features: pd.DataFrame, errors: np.ndarray) -> list[str]:
    """Choose one or two original columns, without dataset-specific names."""
    numeric = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical = [column for column in features.columns if column not in set(numeric)]
    if numeric and categorical:
        return [
            _rank_numeric(features, numeric, errors)[0],
            _rank_categorical(features, categorical, errors)[0],
        ]
    if numeric:
        return _rank_numeric(features, numeric, errors)[:2]
    return _rank_categorical(features, categorical, errors)[:2]


def _rank_numeric(
    features: pd.DataFrame,
    columns: list[Any],
    errors: np.ndarray,
) -> list[str]:
    ranked = sorted(
        columns,
        key=lambda column: (
            -_numeric_association(features[column], errors),
            str(column),
        ),
    )
    return [str(column) for column in ranked]


def _rank_categorical(
    features: pd.DataFrame,
    columns: list[Any],
    errors: np.ndarray,
) -> list[str]:
    ranked = sorted(
        columns,
        key=lambda column: (
            -_categorical_spread(features[column], errors),
            str(column),
        ),
    )
    return [str(column) for column in ranked]


def _numeric_association(series: pd.Series, errors: np.ndarray) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return 0.0
    x_values = values[valid]
    y_values = errors[valid].astype(float)
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return 0.0
    correlation = np.corrcoef(x_values, y_values)[0, 1]
    if not np.isfinite(correlation):
        return 0.0
    return abs(float(correlation))


def _categorical_spread(series: pd.Series, errors: np.ndarray) -> float:
    rows, _omitted, _min_count = _categorical_rows(series, errors)
    if len(rows) < 2:
        return 0.0
    rates = [row["error_rate"] for row in rows]
    return max(rates) - min(rates)


def _slice_feature(series: pd.Series, errors: np.ndarray) -> dict[str, Any]:
    if pd.api.types.is_numeric_dtype(series):
        return _numeric_slice(series, errors)
    return _categorical_slice(series, errors)


def _numeric_slice(series: pd.Series, errors: np.ndarray) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    work = pd.DataFrame({"value": values, "error": errors})
    known = work[np.isfinite(work["value"].to_numpy())].copy()
    bins: list[dict[str, Any]] = []
    if not known.empty and known["value"].nunique(dropna=True) >= 2:
        quantile_count = min(_NUMERIC_BINS, int(known["value"].nunique(dropna=True)))
        known["bin"] = pd.qcut(known["value"], q=quantile_count, duplicates="drop")
        grouped = list(known.groupby("bin", observed=True))
        grouped.sort(key=lambda item: _bin_sort_key(item[0]))
        bins.extend(_bin_record(str(label), group) for label, group in grouped)
    elif not known.empty:
        bins.append(_bin_record("all", known))
    missing = work[~np.isfinite(work["value"].to_numpy())]
    if not missing.empty:
        bins.append(_bin_record("(missing)", missing))
    return {"type": "numeric", "bins": bins}


def _categorical_slice(series: pd.Series, errors: np.ndarray) -> dict[str, Any]:
    rows, omitted, min_count = _categorical_rows(series, errors)
    reported = rows
    truncated = False
    if len(reported) > _MAX_REPORTED_CATEGORIES:
        reported = sorted(reported, key=lambda row: (-row["count"], row["category"]))
        reported = reported[:_MAX_REPORTED_CATEGORIES]
        truncated = True
    reported = sorted(reported, key=lambda row: (-row["error_rate"], row["category"]))
    payload: dict[str, Any] = {
        "type": "categorical",
        "min_count": min_count,
        "omitted_small_categories": omitted,
        "categories": reported,
    }
    if truncated:
        payload["truncated_to"] = _MAX_REPORTED_CATEGORIES
    return payload


def _categorical_rows(
    series: pd.Series,
    errors: np.ndarray,
) -> tuple[list[dict[str, Any]], int, int]:
    labels = [_category_label(value) for value in series.to_numpy()]
    work = pd.DataFrame({"category": labels, "error": errors})
    min_count = _min_category_count(len(work))
    rows = []
    omitted = 0
    for category, group in work.groupby("category", sort=True):
        count = int(len(group))
        if count < min_count:
            omitted += 1
            continue
        n_errors = int(group["error"].sum())
        rows.append(
            {
                "category": str(category),
                "count": count,
                "errors": n_errors,
                "error_rate": _unit_float(n_errors / count),
            }
        )
    return rows, omitted, min_count


def _bin_record(label: str, group: pd.DataFrame) -> dict[str, Any]:
    count = int(len(group))
    n_errors = int(group["error"].sum())
    return {
        "bin": label,
        "count": count,
        "errors": n_errors,
        "error_rate": _unit_float(n_errors / count),
    }


def _bin_sort_key(label: Any) -> tuple:
    if isinstance(label, pd.Interval):
        return (0, float(label.left), float(label.right))
    return (1, str(label))


def _min_category_count(n_samples: int) -> int:
    if n_samples < 50:
        return 1
    return max(20, int(0.005 * n_samples))


def _category_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "(missing)"
    return str(value)


def _probabilities_for_rows(
    model,
    features,
    predictions: np.ndarray,
) -> tuple[list[float | None], list[float | None]]:
    n_rows = predictions.shape[0]
    predicted_probability: list[float | None] = [None] * n_rows
    positive_probability: list[float | None] = [None] * n_rows
    loaded = _probability_matrix(model, features)
    if loaded is None:
        return predicted_probability, positive_probability
    probabilities, class_labels = loaded
    label_to_index = {
        int(label): index for index, label in enumerate(class_labels.tolist())
    }
    positive_index = label_to_index.get(1)
    for position, predicted in enumerate(predictions.tolist()):
        predicted_index = label_to_index.get(int(predicted))
        if predicted_index is not None:
            predicted_probability[position] = _unit_float(
                probabilities[position, predicted_index]
            )
        if positive_index is not None:
            positive_probability[position] = _unit_float(
                probabilities[position, positive_index]
            )
    return predicted_probability, positive_probability


def _probability_matrix(model, features) -> tuple[np.ndarray, np.ndarray] | None:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None
    probabilities = np.asarray(predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "predict_proba() must return one column per class; "
            f"got shape {tuple(probabilities.shape)}."
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Predicted probabilities must be finite and inside [0, 1].")
    classes = getattr(model, "classes_", None)
    if classes is None:
        if probabilities.shape[1] != 2:
            raise ValueError(
                "Cannot choose class probabilities without classes_ when "
                f"predict_proba() has {probabilities.shape[1]} columns."
            )
        class_labels = np.array([0, 1])
    else:
        class_labels = np.asarray(classes)
    return probabilities, class_labels


def _feature_frame(features, feature_names) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if feature_names is not None:
            missing = [name for name in feature_names if name not in frame.columns]
            if missing:
                joined = ", ".join(str(name) for name in missing)
                raise ValueError(f"feature_names not found in X_test: {joined}.")
            frame = frame.loc[:, list(feature_names)].copy()
        frame.columns = [str(column) for column in frame.columns]
        return frame

    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError("X_test must be a 2-dimensional feature matrix.")
    if feature_names is None:
        names = [f"feature_{index}" for index in range(array.shape[1])]
    else:
        names = [str(name) for name in feature_names]
    if len(names) != array.shape[1]:
        raise ValueError(
            f"feature_names has length {len(names)} for {array.shape[1]} columns."
        )
    return pd.DataFrame(array, columns=names)


def _as_binary_labels(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 1:
        array = np.ravel(array)
    try:
        labels = array.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary labels 0 and 1.") from exc
    unexpected = sorted(set(np.unique(labels).tolist()) - {0, 1})
    if unexpected:
        raise ValueError(f"{name} must contain binary labels 0 and 1, found {unexpected}.")
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


def _json_index(value: Any) -> int | float | str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return int(number) if number.is_integer() else number
    return str(value)


def _json_feature_value(value: Any, integer_column: bool) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if integer_column or isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    return str(value)


def _sample_sort_key(sample: dict[str, Any]) -> tuple:
    index = sample["sample_index"]
    if isinstance(index, (int, float)) and not isinstance(index, bool):
        return (0, float(index), sample["test_position"])
    return (1, str(index), sample["test_position"])


def _unit_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"Rate {value!r} is outside [0, 1].")
    return number


def _write_report(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
