"""Measure how Gaussian noise on numerical inputs changes a trained model.

The model is not fit again. Baseline scores come from ``predict`` on the
original test rows. Each later score comes from ``predict`` on a copy whose
numerical columns have been shifted by Gaussian noise. A noise level of
``0.10`` means each selected column is shifted by ``0.10`` times that
column's sample standard deviation. Categorical columns are copied unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.evaluate import LABEL_METRIC_NAMES, score_predictions

DEFAULT_NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.30)
DEFAULT_N_SEEDS = 5
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class RobustnessResult:
    """Baseline metrics and per-noise-level means, spreads, and drops."""

    dataset: str | None
    model: str | None
    baseline_metrics: dict[str, float]
    noise_configuration: dict[str, Any]
    results: tuple[dict[str, Any], ...]
    table: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready robustness report for one model."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "baseline_metrics": dict(self.baseline_metrics),
            "noise_configuration": _json_configuration(self.noise_configuration),
            "results": [_json_level(level) for level in self.results],
        }


def run_robustness(
    model,
    X_test: pd.DataFrame,
    y_test,
    *,
    dataset: str | None = None,
    model_name: str | None = None,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    n_seeds: int = DEFAULT_N_SEEDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    seeds: Sequence[int] | None = None,
) -> RobustnessResult:
    """Score one trained model on noisy copies of ``X_test``.

    ``features`` chooses which numerical columns receive noise. When it is
    omitted, every numerical column is perturbed. ``seeds`` is the explicit
    seed list. When it is omitted, the seeds are
    ``random_state, random_state + 1, ...`` for ``n_seeds`` values.
    """
    if not hasattr(model, "predict"):
        raise TypeError("model must provide predict(). Robustness testing does not fit a model.")
    frame = _require_frame(X_test)
    labels = _aligned_labels(y_test, len(frame))
    levels = _require_noise_levels(noise_levels)
    resolved_seeds, resolved_state = _resolve_seeds(seeds, n_seeds, random_state)
    numeric_columns, categorical_columns = _column_roles(
        frame,
        numeric_features,
        categorical_features,
    )
    perturbed_columns = _selected_numeric_features(frame, numeric_columns, features)
    baseline_predictions = np.asarray(model.predict(frame.copy()))
    baseline = score_predictions(labels, baseline_predictions)
    level_results = tuple(
        _score_noise_level(
            model,
            frame,
            labels,
            baseline,
            noise_level=level,
            selected_features=perturbed_columns,
            seeds=resolved_seeds,
        )
        for level in levels
    )
    configuration = {
        "noise_levels": levels,
        "n_seeds": len(resolved_seeds),
        "seeds": resolved_seeds,
        "random_state": resolved_state,
        "numeric_features": tuple(numeric_columns),
        "categorical_features": tuple(categorical_columns),
        "perturbed_features": tuple(perturbed_columns),
        "noise_scale": "proportion of feature sample standard deviation",
    }
    return RobustnessResult(
        dataset=dataset,
        model=model_name,
        baseline_metrics=baseline,
        noise_configuration=configuration,
        results=level_results,
        table=_result_table(dataset, model_name, baseline, level_results),
    )


def perturb_features(
    features: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    """Return a new frame with Gaussian noise on ``numeric_features``.

    ``features`` is not modified. Missing numerical values stay missing.
    Columns that are not listed are copied unchanged. ``noise_level`` is the
    fraction of each column's sample standard deviation used as the Gaussian
    scale. A level of ``0`` returns an exact copy.
    """
    frame = _require_frame(features)
    level = _require_noise_levels((noise_level,))[0]
    resolved_seed = _require_seed(seed, "seed")
    missing = [str(name) for name in numeric_features if str(name) not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Numerical features were not found in the test data: {joined}.")
    perturbed = frame.copy()
    if level == 0.0 or len(numeric_features) == 0:
        return perturbed
    generator = np.random.default_rng(resolved_seed)
    for name in numeric_features:
        column_name = str(name)
        series = perturbed[column_name]
        if not _is_numeric_series(series):
            raise ValueError(
                "Gaussian noise is only applied to numerical features. "
                f"Non-numerical feature: {column_name}."
            )
        perturbed[column_name] = _perturb_column(
            series,
            noise_level=level,
            generator=generator,
        )
    return perturbed


def compare_robustness(results: Sequence[RobustnessResult]) -> pd.DataFrame:
    """Stack per-model robustness tables for a later comparison."""
    if isinstance(results, RobustnessResult):
        raise TypeError("compare_robustness expects a sequence of RobustnessResult values.")
    if len(results) == 0:
        raise ValueError("compare_robustness needs at least one robustness result.")
    frames = []
    for result in results:
        if not isinstance(result, RobustnessResult):
            raise TypeError("compare_robustness expects RobustnessResult values.")
        frames.append(result.table)
    return pd.concat(frames, ignore_index=True)


def compare_robustness_files(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Compare robustness JSON files already written for one or more models."""
    if isinstance(paths, (str, Path)):
        raise TypeError("compare_robustness_files expects a sequence of JSON paths.")
    if len(paths) == 0:
        raise ValueError("compare_robustness_files needs at least one robustness file.")
    frames = [_table_from_payload(_read_robustness_payload(path)) for path in paths]
    return pd.concat(frames, ignore_index=True)


def save_robustness_result(result: RobustnessResult, directory: Path | str) -> Path:
    """Write ``<dataset>_<model>_robustness.json`` under ``directory``."""
    if not result.dataset or not result.model:
        raise ValueError("Saving robustness results requires both a dataset name and a model name.")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.dataset}_{result.model}_robustness.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def robustness_directory(model_dir: Path | str) -> Path:
    """Return ``outputs/diagnostics/robustness`` for a normal experiment directory."""
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "robustness"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "robustness"
    return directory / "diagnostics" / "robustness"


def _score_noise_level(
    model,
    frame: pd.DataFrame,
    labels,
    baseline: dict[str, float],
    *,
    noise_level: float,
    selected_features: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        perturbed = perturb_features(
            frame,
            numeric_features=selected_features,
            noise_level=noise_level,
            seed=seed,
        )
        predictions = np.asarray(model.predict(perturbed))
        metrics = score_predictions(labels, predictions)
        runs.append({"seed": int(seed), **metrics})
    mean = {}
    std = {}
    degradation = {}
    for name in LABEL_METRIC_NAMES:
        values = [run[name] for run in runs]
        mean[name], std[name] = _mean_std(values)
        degradation[name] = float(baseline[name]) - mean[name]
    return {
        "noise_level": float(noise_level),
        "mean": mean,
        "std": std,
        "degradation": degradation,
        "runs": tuple(runs),
    }


def _perturb_column(series: pd.Series, *, noise_level: float, generator) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    observed = int(finite.sum())
    if observed < 2:
        scale = 0.0
    else:
        scale = float(np.std(values[finite], ddof=1))
    if scale == 0.0 or noise_level == 0.0:
        return series.copy()
    noise = generator.normal(loc=0.0, scale=noise_level * scale, size=values.shape[0])
    updated = values.copy()
    updated[finite] = updated[finite] + noise[finite]
    return pd.Series(updated, index=series.index, name=series.name)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("A noise level needs at least one seed.")
    if np.all(array == array[0]):
        return float(array[0]), 0.0
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0
    return mean, float(np.std(array, ddof=1))


def _read_robustness_payload(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Robustness file not found: {source}.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot read {source.name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source.name} must contain a JSON object.")
    missing = [
        name
        for name in ("dataset", "model", "baseline_metrics", "results")
        if name not in payload
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source.name} is missing {joined}.")
    return payload


def _table_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    baseline = payload["baseline_metrics"]
    rows = []
    for level in payload["results"]:
        row = {
            "dataset": payload["dataset"],
            "model": payload["model"],
            "noise_level": level["noise_level"],
        }
        for name in LABEL_METRIC_NAMES:
            row[f"{name}_baseline"] = baseline[name]
            row[f"{name}_mean"] = level["mean"][name]
            row[f"{name}_std"] = level["std"][name]
            row[f"{name}_drop"] = level["degradation"][name]
        rows.append(row)
    if not rows:
        raise ValueError(
            f"Robustness results for {payload['model']!r} do not contain a noise level."
        )
    return pd.DataFrame.from_records(rows)


def _result_table(dataset, model_name, baseline, levels) -> pd.DataFrame:
    rows = []
    for level in levels:
        row = {
            "dataset": dataset,
            "model": model_name,
            "noise_level": level["noise_level"],
        }
        for name in LABEL_METRIC_NAMES:
            row[f"{name}_baseline"] = baseline[name]
            row[f"{name}_mean"] = level["mean"][name]
            row[f"{name}_std"] = level["std"][name]
            row[f"{name}_drop"] = level["degradation"][name]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _column_roles(features: pd.DataFrame, numeric_features, categorical_features):
    columns = [str(column) for column in features.columns]
    numeric_override = _name_list(numeric_features, "numeric_features")
    categorical_override = _name_list(categorical_features, "categorical_features")
    if numeric_override is not None:
        _require_known_columns(features, numeric_override, "Numerical features")
        numeric = numeric_override
    else:
        excluded = set(categorical_override or [])
        numeric = [
            column
            for column in columns
            if column not in excluded and _is_numeric_series(features[column])
        ]
    if categorical_override is not None:
        _require_known_columns(features, categorical_override, "Categorical features")
        categorical = categorical_override
    else:
        numeric_names = set(numeric)
        categorical = [column for column in columns if column not in numeric_names]
    overlap = sorted(set(numeric).intersection(categorical))
    if overlap:
        joined = ", ".join(overlap)
        raise ValueError(f"Features cannot be both numerical and categorical: {joined}.")
    return numeric, categorical


def _selected_numeric_features(features, numeric_columns, selected) -> list[str]:
    if selected is None:
        return list(numeric_columns)
    names = _name_list(selected, "features")
    _require_known_columns(features, names, "Selected features")
    numeric_names = set(numeric_columns)
    categorical = [name for name in names if name not in numeric_names]
    if categorical:
        joined = ", ".join(categorical)
        raise ValueError(
            "Gaussian noise is only applied to numerical features. "
            f"Categorical features were selected: {joined}."
        )
    return names


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    return bool(pd.api.types.is_numeric_dtype(series))


def _require_frame(features) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Test features must be a DataFrame.")
    if features.empty:
        raise ValueError("Test features must contain at least one row.")
    return features


def _aligned_labels(y_test, n_rows: int):
    labels = np.asarray(y_test)
    if labels.shape[0] != n_rows:
        raise ValueError(
            "y_test and X_test have different lengths: "
            f"{labels.shape[0]} and {n_rows}."
        )
    return labels


def _require_noise_levels(noise_levels) -> tuple[float, ...]:
    if isinstance(noise_levels, (str, bytes)) or not isinstance(noise_levels, Sequence):
        raise TypeError("noise_levels must be a sequence of proportions.")
    if len(noise_levels) == 0:
        raise ValueError("noise_levels must contain at least one level.")
    parsed = []
    for level in noise_levels:
        if isinstance(level, bool) or not isinstance(level, (int, float, np.floating)):
            raise TypeError("Each noise level must be a number.")
        value = float(level)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Each noise level must be finite and at least 0.")
        parsed.append(value)
    return tuple(parsed)


def _resolve_seeds(seeds, n_seeds: int, random_state: int) -> tuple[tuple[int, ...], int]:
    state = _require_seed(random_state, "random_state")
    if seeds is None:
        count = _require_count(n_seeds, "n_seeds")
        return tuple(range(state, state + count)), state
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of integers.")
    parsed = tuple(_require_seed(seed, "seed") for seed in seeds)
    if len(parsed) == 0:
        raise ValueError("seeds must contain at least one seed.")
    return parsed, state


def _require_seed(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _require_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(value)


def _name_list(names, label: str) -> list[str] | None:
    if names is None:
        return None
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(f"{label} must be a sequence of column names.")
    return [str(name) for name in names]


def _require_known_columns(features: pd.DataFrame, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if name not in features.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} were not found in the test data: {joined}.")


def _json_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    payload = dict(configuration)
    for key in ("noise_levels", "seeds", "numeric_features", "categorical_features", "perturbed_features"):
        payload[key] = list(payload[key])
    return payload


def _json_level(level: dict[str, Any]) -> dict[str, Any]:
    return {
        "noise_level": level["noise_level"],
        "mean": dict(level["mean"]),
        "std": dict(level["std"]),
        "degradation": dict(level["degradation"]),
        "runs": [dict(run) for run in level["runs"]],
    }
