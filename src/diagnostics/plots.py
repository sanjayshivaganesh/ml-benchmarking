"""Matplotlib charts for stored Phase 3 diagnostic results.

The plot functions draw stored slice F1, robustness, feature importance, and
confidence values. They do not train a model or recompute metrics. Saving
plots is optional. The diagnostic engine and the JSON and markdown reports
do not call this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.diagnostics.diagnostic_engine import DiagnosticEngineResult
from src.diagnostics.prediction_store import prediction_directory, prediction_filename

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIRECTORY = _PROJECT_ROOT / "outputs" / "diagnostics" / "plots"
_CONFIDENCE_BINS = 10


def plot_slice_f1(
    slices: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    model: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw stored F1 for each feature slice."""
    labels, values = _slice_f1(slices)
    axis, created = _axis(ax)
    try:
        axis.barh(labels, values)
        axis.set_xlabel("F1")
        axis.set_ylabel("Feature slice")
        axis.set_xlim(0.0, 1.0)
        axis.set_title(_title("Slice F1", dataset, model))
        axis.invert_yaxis()
    except Exception:
        _close_if_created(axis, created)
        raise
    return axis


def plot_robustness_curve(
    levels: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    model: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw stored mean F1 against noise level for one model."""
    noise, scores = _robustness_points(levels)
    axis, created = _axis(ax)
    try:
        axis.plot(noise, scores, marker="o")
        axis.set_xlabel("Noise level")
        axis.set_ylabel("F1")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(_title("Robustness", dataset, model))
    except Exception:
        _close_if_created(axis, created)
        raise
    return axis


def plot_robustness_comparison(
    series: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw one stored robustness curve for each model."""
    drawn = _comparison_series(series)
    axis, created = _axis(ax)
    try:
        for model, noise, scores in drawn:
            axis.plot(noise, scores, marker="o", label=str(model))
        axis.set_xlabel("Noise level")
        axis.set_ylabel("F1")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(_title("Robustness comparison", dataset, None))
        axis.legend()
    except Exception:
        _close_if_created(axis, created)
        raise
    return axis


def plot_feature_importance(
    features: Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    model: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw stored feature-importance ranks."""
    labels, values = _importance_bars(features)
    axis, created = _axis(ax)
    try:
        axis.barh(labels, values)
        axis.set_xlabel("Importance")
        axis.set_ylabel("Feature")
        axis.set_title(_title("Feature importance", dataset, model))
        axis.invert_yaxis()
    except Exception:
        _close_if_created(axis, created)
        raise
    return axis


def plot_confidence_distribution(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    model: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw stored confidence separately for correct and incorrect predictions."""
    correct, incorrect, bins = _confidence_groups(records)
    axis, created = _axis(ax)
    try:
        if correct.size:
            axis.hist(correct, bins=bins, histtype="step", label="correct")
        if incorrect.size:
            axis.hist(incorrect, bins=bins, histtype="step", label="incorrect")
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Count")
        axis.set_title(_title("Confidence", dataset, model))
        axis.legend()
    except Exception:
        _close_if_created(axis, created)
        raise
    return axis


def save_diagnostic_plots(
    result: DiagnosticEngineResult | dict[str, Any],
    directory: Path | str | None = None,
    predictions: Mapping[tuple[str, str], pd.DataFrame | Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Path]:
    """Write the plots that have stored data.

    A model without slice F1, a robustness curve, feature importance, or
    prediction rows does not produce that file. The returned mapping contains
    only the files that were written.
    """
    payload = _payload(result)
    destination = Path(directory) if directory is not None else _plots_directory(payload)
    written: dict[str, Path] = {}
    by_dataset: dict[Any, list[dict[str, Any]]] = {}
    for run in _runs(payload):
        dataset, model = _names(run)
        by_dataset.setdefault(dataset, []).append(run)
        _write_plot(
            written,
            f"{dataset}_{model}_slice_f1",
            destination / f"{dataset}_{model}_slice_f1.png",
            lambda run=run: plot_slice_f1(
                _stored(run, "slicing").get("slices") or [],
                dataset=dataset,
                model=model,
            ),
        )
        _write_plot(
            written,
            f"{dataset}_{model}_robustness_curve",
            destination / f"{dataset}_{model}_robustness_curve.png",
            lambda run=run: plot_robustness_curve(
                _stored(run, "robustness").get("results") or [],
                dataset=dataset,
                model=model,
            ),
        )
        _write_plot(
            written,
            f"{dataset}_{model}_feature_importance",
            destination / f"{dataset}_{model}_feature_importance.png",
            lambda run=run: plot_feature_importance(
                _importance_rows(run),
                dataset=dataset,
                model=model,
            ),
        )
        _write_plot(
            written,
            f"{dataset}_{model}_confidence_distribution",
            destination / f"{dataset}_{model}_confidence_distribution.png",
            lambda run=run: plot_confidence_distribution(
                _prediction_rows(run, predictions),
                dataset=dataset,
                model=model,
            ),
        )
    for dataset, runs in by_dataset.items():
        _write_plot(
            written,
            f"{dataset}_robustness_comparison",
            destination / f"{dataset}_robustness_comparison.png",
            lambda dataset=dataset, runs=runs: plot_robustness_comparison(
                [
                    {
                        "model": _names(run)[1],
                        "results": _stored(run, "robustness").get("results") or [],
                    }
                    for run in runs
                ],
                dataset=dataset,
            ),
        )
    return written


def _write_plot(written: dict[str, Path], key: str, path: Path, draw) -> None:
    try:
        axis = draw()
    except ValueError:
        return
    figure = axis.figure
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)
    written[key] = path


def _slice_f1(slices: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for item in slices:
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
        value = _number(metrics.get("f1"))
        feature = item.get("feature")
        definition = item.get("slice_definition")
        if value is None or feature is None or definition is None:
            continue
        labels.append(f"{feature}: {definition}")
        values.append(value)
    if not values:
        raise ValueError("No slice F1 values are available to plot.")
    return labels, values


def _robustness_points(levels: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    points = []
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        noise = _number(level.get("noise_level"))
        mean = level.get("mean") if isinstance(level.get("mean"), Mapping) else {}
        score = _number(mean.get("f1"))
        if noise is None or score is None:
            continue
        points.append((noise, score))
    if not points:
        raise ValueError("No robustness F1 values are available to plot.")
    points.sort(key=lambda item: (item[0], item[1]))
    return [item[0] for item in points], [item[1] for item in points]


def _comparison_series(series: Sequence[Mapping[str, Any]]) -> list[tuple[Any, list[float], list[float]]]:
    drawn = []
    for item in series:
        if not isinstance(item, Mapping) or item.get("model") is None:
            continue
        try:
            noise, scores = _robustness_points(item.get("results") or [])
        except ValueError:
            continue
        drawn.append((item.get("model"), noise, scores))
    if len(drawn) < 2:
        raise ValueError("Robustness comparison needs stored curves for at least two models.")
    return drawn


def _importance_bars(features: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for item in features:
        if not isinstance(item, Mapping):
            continue
        value = _number(item.get("importance"))
        feature = item.get("feature")
        if value is None or not feature:
            continue
        labels.append(str(feature))
        values.append(value)
    if not values:
        raise ValueError("No feature-importance values are available to plot.")
    return labels, values


def _importance_rows(run: dict[str, Any]) -> list[Mapping[str, Any]]:
    component = run.get("explanation")
    if not isinstance(component, Mapping) or component.get("status") != "supported":
        return []
    stored = _stored(run, "explanation")
    features = stored.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return []
    return [item for item in features if isinstance(item, Mapping)]


def _confidence_groups(records) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _confidence_frame(records)
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(dtype=float)
    correct = _correct_flags(frame["correct"])
    finite = np.isfinite(confidence)
    confidence = confidence[finite]
    correct = correct[finite]
    if confidence.size == 0:
        raise ValueError("No confidence values are available to plot.")
    correct_values = confidence[correct]
    incorrect_values = confidence[~correct]
    low = min(0.5, float(np.min(confidence)))
    high = max(1.0, float(np.max(confidence)))
    if low == high:
        high = low + 1.0
    return correct_values, incorrect_values, np.linspace(low, high, _CONFIDENCE_BINS + 1)


def _confidence_frame(records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        rows = [dict(item) for item in records if isinstance(item, Mapping)]
        frame = pd.DataFrame(rows)
    missing = [name for name in ("confidence", "correct") if name not in frame.columns]
    if missing or frame.empty:
        raise ValueError("No confidence values are available to plot.")
    return frame


def _correct_flags(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float) != 0.0
    text = values.astype(str).str.strip().str.lower()
    return text.isin(("true", "1", "1.0")).to_numpy()


def _prediction_rows(run, predictions):
    dataset, model = _names(run)
    if predictions is not None:
        stored = predictions.get((str(dataset), str(model)))
        if stored is not None:
            return stored
    experiment = run.get("experiment") if isinstance(run.get("experiment"), Mapping) else {}
    experiment_dir = experiment.get("experiment_dir")
    if experiment_dir is None or dataset is None or model is None:
        raise ValueError("No confidence values are available to plot.")
    path = prediction_directory(experiment_dir) / prediction_filename(str(dataset), str(model))
    if not path.is_file():
        raise ValueError("No confidence values are available to plot.")
    return pd.read_csv(path)


def _stored(run: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    component = run.get(name)
    if not isinstance(component, Mapping) or component.get("status") != "supported":
        return {}
    result = component.get("result")
    if isinstance(result, Mapping):
        return result
    return {}


def _runs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [run for run in payload.get("runs") or [] if isinstance(run, dict)]


def _names(run: Mapping[str, Any]) -> tuple[Any, Any]:
    experiment = run.get("experiment") if isinstance(run.get("experiment"), Mapping) else {}
    return experiment.get("dataset"), experiment.get("model")


def _plots_directory(payload: Mapping[str, Any]) -> Path:
    for run in _runs(payload):
        experiment = run.get("experiment") if isinstance(run.get("experiment"), Mapping) else {}
        experiment_dir = experiment.get("experiment_dir")
        if experiment_dir:
            return _directory_for_experiment(experiment_dir)
    return DEFAULT_PLOTS_DIRECTORY


def _directory_for_experiment(model_dir: Path | str) -> Path:
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "plots"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "plots"
    return directory / "diagnostics" / "plots"


def _payload(result: DiagnosticEngineResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, DiagnosticEngineResult):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError("Diagnostic plots expect a DiagnosticEngineResult or its dictionary.")


def _axis(ax: Axes | None) -> tuple[Axes, bool]:
    if ax is not None:
        return ax, False
    _figure, axis = plt.subplots()
    return axis, True


def _close_if_created(axis: Axes, created: bool) -> None:
    if created:
        plt.close(axis.figure)


def _title(label: str, dataset: str | None, model: str | None) -> str:
    if dataset and model:
        return f"{label}: {dataset} / {model}"
    if dataset:
        return f"{label}: {dataset}"
    if model:
        return f"{label}: {model}"
    return label


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return number
    return None
