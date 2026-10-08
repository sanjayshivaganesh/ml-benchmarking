"""Standard prediction rows for later diagnostic modules.

Phase 2 already stores ``predictions`` and ``positive_class_probabilities``
on each fitted model, with sample ids on ``y_test.index`` and in
``predictions.json`` as ``test_index``. This module turns those values into
one table. It does not train, score, or slice errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "sample_id",
    "y_true",
    "y_pred",
    "probability",
    "confidence",
    "correct",
)

_PROBABILITY_ATOL = 1e-8


def prediction_filename(dataset: str, model: str) -> str:
    """Return ``<dataset>_<model>_predictions.csv``."""
    return f"{dataset}_{model}_predictions.csv"


def prediction_directory(model_dir: Path | str) -> Path:
    """Return the directory that receives prediction CSVs for one run.

    A model directory of ``outputs/experiments`` or
    ``outputs/experiments/<experiment_id>`` writes to ``outputs/predictions``.
    Any other directory keeps its CSVs in a ``predictions`` folder inside it,
    so a temporary experiment does not write into the project tree.
    """
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "predictions"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "predictions"
    return directory / "predictions"


def save_experiment_predictions(result, model_dir: Path | str) -> dict[str, Path]:
    """Write one prediction CSV per model already stored on ``result``.

    This reads ``predictions`` and ``positive_class_probabilities``. It does
    not call ``predict`` or recompute metrics. ``sample_id`` is ``y_test.index``,
    so the original ``X_test`` rows can be joined later.
    """
    written: dict[str, Path] = {}
    destination = prediction_directory(model_dir)
    for model_name in result.models:
        store = PredictionStore.from_experiment(result, model_name)
        path = destination / prediction_filename(result.dataset, model_name)
        written[model_name] = store.save_csv(path)
    return written


class PredictionStore:
    """One model's predictions, aligned to the original sample ids.

    ``frame`` always has ``PREDICTION_COLUMNS``. ``features`` is the original
    feature rows indexed by ``sample_id`` when the caller supplied them.
    Later code can also join those rows with ``join_features``.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> None:
        validated = _validate_frame(frame)
        self.frame = validated
        self.features = _stored_features(features, validated["sample_id"])

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        probability,
        *,
        sample_ids=None,
        features: pd.DataFrame | None = None,
        classes=None,
    ) -> PredictionStore:
        """Build a store from labels, predictions, and probabilities.

        A one-dimensional ``probability`` is the positive-class probability
        for binary labels ``{0, 1}``. Confidence is ``max(p, 1 - p)``.

        A two-dimensional ``probability`` with classes ``{0, 1}`` uses the
        column for class ``1`` the same way. Any other class set is
        multiclass: ``probability`` is the probability of ``y_pred``, and
        ``confidence`` is the largest class probability.

        ``sample_ids`` override the index of a ``y_true`` Series. Without
        either one, ids are ``0 .. n-1``.
        """
        true_labels = _as_label_array(y_true, "y_true")
        predicted = _as_label_array(y_pred, "y_pred")
        _require_same_length(true_labels, predicted, "y_true", "y_pred")
        ids = _sample_ids(sample_ids, y_true, true_labels.shape[0])
        probability_column, confidence = _probability_and_confidence(
            probability,
            true_labels,
            predicted,
            classes,
        )
        _require_same_length(
            true_labels,
            probability_column,
            "y_true",
            "probability",
        )
        correct = _labels_match(true_labels, predicted)
        frame = pd.DataFrame(
            {
                "sample_id": ids,
                "y_true": true_labels,
                "y_pred": predicted,
                "probability": probability_column,
                "confidence": confidence,
                "correct": correct,
            }
        )
        return cls(frame, features=features)

    @classmethod
    def from_model_result(
        cls,
        model_result,
        y_true,
        *,
        features: pd.DataFrame | None = None,
        sample_ids=None,
    ) -> PredictionStore:
        """Build a store from a Phase 2 model result.

        The result must expose ``predictions`` and
        ``positive_class_probabilities``. ``y_true`` is the experiment
        ``y_test`` series when the original row index should be kept.
        ``features`` is ``X_test`` when the original rows should travel
        with the predictions.
        """
        predictions = getattr(model_result, "predictions", None)
        positive = getattr(model_result, "positive_class_probabilities", None)
        if predictions is None or positive is None:
            raise ValueError(
                "A model result needs predictions and "
                "positive_class_probabilities."
            )
        return cls.from_predictions(
            y_true,
            predictions,
            positive,
            sample_ids=sample_ids,
            features=features,
        )

    @classmethod
    def from_experiment(cls, result, model: str) -> PredictionStore:
        """Build a store for one model on an experiment result.

        Sample ids come from ``y_test.index``. Feature rows come from
        ``X_test``. Both are already on ``ExperimentResult``.
        """
        fitted = getattr(result, "fitted", None)
        if not isinstance(fitted, dict) or model not in fitted:
            available = _available_models(result)
            raise ValueError(
                f"Model {model!r} is not in this experiment. "
                f"Available models: {available}."
            )
        y_true = getattr(result, "y_test", None)
        if y_true is None:
            raise ValueError("The experiment result has no y_test.")
        features = getattr(result, "X_test", None)
        if features is not None and not isinstance(features, pd.DataFrame):
            raise TypeError("Experiment X_test must be a DataFrame when present.")
        return cls.from_model_result(fitted[model], y_true, features=features)

    @classmethod
    def from_phase2_predictions(
        cls,
        payload: dict[str, Any],
        model: str,
        features: pd.DataFrame | None = None,
    ) -> PredictionStore:
        """Build a store from a Phase 2 ``predictions.json`` object.

        ``test_index`` becomes ``sample_id``. Probabilities are the stored
        positive-class probabilities for ``model``.
        """
        if not isinstance(payload, dict):
            raise TypeError("Phase 2 predictions must be a dictionary.")
        missing = [name for name in ("test_index", "y_test", "models") if name not in payload]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Phase 2 predictions are missing {joined}.")
        models = payload["models"]
        if not isinstance(models, dict) or model not in models:
            available = ", ".join(str(name) for name in models) if isinstance(models, dict) else ""
            raise ValueError(
                f"Model {model!r} is not in the saved predictions. "
                f"Available models: {available}."
            )
        entry = models[model]
        if not isinstance(entry, dict):
            raise ValueError(f"Saved predictions for {model!r} must be a dictionary.")
        for name in ("predictions", "positive_class_probabilities"):
            if name not in entry:
                raise ValueError(
                    f"Saved predictions for {model!r} are missing {name}."
                )
        return cls.from_predictions(
            payload["y_test"],
            entry["predictions"],
            entry["positive_class_probabilities"],
            sample_ids=payload["test_index"],
            features=features,
        )

    def to_frame(self, *, include_features: bool = False) -> pd.DataFrame:
        """Return a copy of the prediction table.

        With ``include_features=True``, original feature columns are joined
        on ``sample_id``. The feature rows must already be stored.
        """
        frame = self.frame.copy()
        if not include_features:
            return frame
        if self.features is None:
            raise ValueError(
                "Feature rows were not stored. Join them later with "
                "join_features() using sample_id."
            )
        return _concat_features(frame, self.features)

    def join_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions joined to feature rows on ``sample_id``.

        ``features`` may be indexed by sample id or contain a ``sample_id``
        column. This is the join for an ``X_test`` frame that was not stored
        with the predictions.
        """
        aligned = _align_features(features, self.frame["sample_id"].tolist(), join=True)
        return _concat_features(self.frame, aligned)

    def save_csv(self, path: Path | str) -> Path:
        """Write the standardized prediction columns to CSV."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(destination, index=False)
        return destination

    @classmethod
    def load_csv(cls, path: Path | str) -> PredictionStore:
        """Read a prediction CSV and require the standardized columns.

        Feature rows are not in this file. Join them afterwards on
        ``sample_id``.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Prediction file not found: {source}.")
        try:
            loaded = pd.read_csv(source)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Prediction file {source} is empty.") from exc
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(f"Cannot read predictions from {source}. {exc}") from exc
        return cls(_coerce_loaded_frame(loaded))


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Predictions must be a DataFrame.")
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Prediction rows are missing required columns: {joined}.")
    selected = frame.loc[:, PREDICTION_COLUMNS].copy()
    if selected.empty:
        raise ValueError("Predictions must contain at least one sample.")
    if selected.isna().any().any():
        empty = [column for column in PREDICTION_COLUMNS if selected[column].isna().any()]
        joined = ", ".join(empty)
        raise ValueError(f"Prediction rows have missing values in {joined}.")
    ids = selected["sample_id"]
    if ids.duplicated().any():
        raise ValueError("sample_id values must be unique.")
    probability = _finite_unit_interval(selected["probability"], "probability")
    confidence = _finite_unit_interval(selected["confidence"], "confidence")
    if np.any(confidence + _PROBABILITY_ATOL < probability):
        raise ValueError(
            "confidence must be at least the stored probability for every sample."
        )
    correct = _as_bool_array(selected["correct"], "correct")
    expected = _labels_match(selected["y_true"].to_numpy(), selected["y_pred"].to_numpy())
    if not np.array_equal(correct, expected):
        raise ValueError("correct does not match the comparison of y_true and y_pred.")
    selected["probability"] = probability
    selected["confidence"] = confidence
    selected["correct"] = correct
    selected = selected.reset_index(drop=True)
    return selected


def _stored_features(features: pd.DataFrame | None, sample_ids: pd.Series) -> pd.DataFrame | None:
    if features is None:
        return None
    aligned = _align_features(features, sample_ids.tolist(), join=False)
    return aligned


def _align_features(features, sample_ids: list[Any], *, join: bool) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame.")
    if "sample_id" in features.columns:
        indexed = features.set_index("sample_id", drop=True)
    else:
        indexed = features
    overlap = sorted(set(indexed.columns).intersection(PREDICTION_COLUMNS))
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(
            f"Feature columns use prediction column names: {joined}."
        )
    ids = pd.Index(sample_ids)
    if len(indexed) != len(ids) and join:
        missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
        if missing:
            shown = ", ".join(repr(value) for value in missing[:5])
            raise ValueError(
                "Feature rows do not include every sample_id. "
                f"Missing sample ids include {shown}."
            )
    if indexed.index.has_duplicates:
        raise ValueError("Feature row sample ids must be unique.")
    if indexed.index.equals(ids):
        aligned = indexed.copy()
    elif indexed.index.is_unique and ids.isin(indexed.index).all():
        aligned = indexed.loc[list(ids)].copy()
    elif (not join) and len(indexed) == len(ids):
        aligned = indexed.copy()
        aligned.index = ids
    else:
        raise ValueError(
            "Feature rows cannot be aligned to sample_id. "
            "Index the features by sample id or keep them in the same order."
        )
    aligned.index.name = "sample_id"
    return aligned


def _concat_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_frame = features.reset_index(drop=True)
    combined = pd.concat(
        [frame.reset_index(drop=True), feature_frame],
        axis=1,
    )
    return combined


def _probability_and_confidence(probability, y_true: np.ndarray, y_pred: np.ndarray, classes):
    values = np.asarray(probability, dtype=float)
    if values.ndim == 1:
        _binary_labels(y_true, "y_true")
        positive = _binary_positive_probability(values, y_pred, "probability")
        return positive, np.maximum(positive, 1.0 - positive)
    if values.ndim != 2:
        raise ValueError(
            "probability must be a positive-class vector or a class-probability "
            f"matrix, got {values.ndim} dimensions."
        )
    if values.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "probability and y_pred have different lengths: "
            f"{values.shape[0]} and {y_pred.shape[0]}."
        )
    if values.shape[1] < 2:
        raise ValueError(
            "A class-probability matrix needs at least two classes, "
            f"found {values.shape[1]}."
        )
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            "Class probabilities must be finite and inside [0, 1]."
        )
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Each class-probability row must sum to 1.")
    class_labels = _class_labels(classes, values.shape[1])
    if _is_project_binary(class_labels):
        positive_index = int(np.flatnonzero(np.asarray(class_labels) == 1)[0])
        positive = values[:, positive_index]
        _binary_labels(y_true, "y_true")
        _binary_labels(y_pred, "y_pred")
        return positive, np.maximum(positive, 1.0 - positive)
    indexes = _predicted_class_indexes(y_pred, class_labels)
    predicted_probability = values[np.arange(values.shape[0]), indexes]
    confidence = values.max(axis=1)
    return predicted_probability, confidence


def _binary_positive_probability(values: np.ndarray, y_pred: np.ndarray, name: str) -> np.ndarray:
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            f"{name} must be a finite positive-class probability inside [0, 1]."
        )
    _binary_labels(y_pred, "y_pred")
    return values.astype(float, copy=False)


def _class_labels(classes, n_columns: int) -> np.ndarray:
    if classes is None:
        return np.arange(n_columns)
    labels = np.asarray(classes)
    if labels.ndim != 1 or labels.shape[0] != n_columns:
        raise ValueError(
            "classes must contain one label per probability column, "
            f"found {labels.tolist() if labels.ndim == 1 else labels.shape}."
        )
    if pd.Index(labels).has_duplicates:
        raise ValueError("classes must not contain duplicate labels.")
    return labels


def _is_project_binary(class_labels: np.ndarray) -> bool:
    if class_labels.shape[0] != 2:
        return False
    return set(_label_key(label) for label in class_labels.tolist()) == {0, 1}


def _predicted_class_indexes(y_pred: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    lookup = {_label_key(label): index for index, label in enumerate(class_labels.tolist())}
    indexes = []
    for label in y_pred.tolist():
        key = _label_key(label)
        if key not in lookup:
            known = ", ".join(repr(value) for value in class_labels.tolist())
            raise ValueError(
                f"Predicted label {label!r} is not in classes {known}."
            )
        indexes.append(lookup[key])
    return np.asarray(indexes, dtype=int)


def _binary_labels(values: np.ndarray, name: str) -> None:
    keys = {_label_key(value) for value in values.tolist()}
    if not keys <= {0, 1}:
        raise ValueError(
            f"{name} must use binary labels 0 and 1 for a positive-class probability."
        )


def _sample_ids(sample_ids, y_true, n_rows: int) -> np.ndarray:
    if sample_ids is None:
        if isinstance(y_true, pd.Series):
            ids = np.asarray(y_true.index.tolist())
        else:
            ids = np.arange(n_rows)
    else:
        ids = np.asarray(list(sample_ids) if not isinstance(sample_ids, np.ndarray) else sample_ids)
        if ids.ndim > 1:
            ids = np.ravel(ids)
    if ids.shape[0] != n_rows:
        raise ValueError(
            "sample_id and y_true have different lengths: "
            f"{ids.shape[0]} and {n_rows}."
        )
    if pd.isna(pd.Series(ids)).any():
        raise ValueError("sample_id values must not be missing.")
    if pd.Index(ids).has_duplicates:
        raise ValueError("sample_id values must be unique.")
    return ids


def _as_label_array(values, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        array = values.to_numpy()
    else:
        array = np.asarray(values)
    if array.dtype == bool:
        raise TypeError(f"{name} must be class labels, not booleans.")
    if array.ndim > 1:
        array = np.ravel(array)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if pd.isna(pd.Series(array)).any():
        raise ValueError(f"{name} must not contain missing labels.")
    return array


def _require_same_length(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> None:
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"{left_name} and {right_name} have different lengths: "
            f"{left.shape[0]} and {right.shape[0]}."
        )


def _labels_match(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_label_key(left) == _label_key(right) for left, right in zip(y_true.tolist(), y_pred.tolist())],
        dtype=bool,
    )


def _label_key(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Class labels must not be booleans.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"Class label {value!r} is not an integer label.")
        return int(value)
    if isinstance(value, str):
        return value
    return value


def _finite_unit_interval(values: pd.Series, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be finite and inside [0, 1].")
    return array


def _as_bool_array(values: pd.Series, name: str) -> np.ndarray:
    converted = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            converted.append(bool(int(value)))
            continue
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            converted.append(value.strip().lower() == "true")
            continue
        raise ValueError(f"{name} must be a boolean, found {value!r}.")
    return np.asarray(converted, dtype=bool)


def _coerce_loaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Prediction rows are missing required columns: {joined}.")
    selected = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    selected["correct"] = _as_bool_array(selected["correct"], "correct")
    return selected


def _available_models(result) -> str:
    models = getattr(result, "models", None)
    if models:
        return ", ".join(str(name) for name in models)
    fitted = getattr(result, "fitted", None)
    if isinstance(fitted, dict):
        return ", ".join(str(name) for name in fitted)
    return ""
