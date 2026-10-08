"""Compare stored diagnostics for the models in one dataset.

Every value is copied from a diagnostic result. A missing measurement stays
missing. Rankings use those stored numbers only, and a ranking is reported
only when at least two models have the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.diagnostic_engine import DiagnosticEngineResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_COMPARISON_PATH = _PROJECT_ROOT / "outputs" / "diagnostics" / "model_comparison.json"
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_MORE_IMPORTANT_WITHIN_MODEL = "higher_is_more_important_within_model"
_MIN_MODELS = 2
_TOO_FEW = (
    "Fewer than two models have this stored measurement, "
    "so no comparison is identified."
)


def build_model_comparison(result: DiagnosticEngineResult | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready comparison of models within each dataset."""
    payload = _payload(result)
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for run in payload.get("runs") or []:
        if not isinstance(run, dict):
            continue
        experiment = run.get("experiment") if isinstance(run.get("experiment"), dict) else {}
        grouped.setdefault(experiment.get("dataset"), []).append(run)
    return {
        "comparisons": [
            _dataset_comparison(dataset, runs) for dataset, runs in grouped.items()
        ]
    }


def save_model_comparison(
    result: DiagnosticEngineResult | dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Write ``model_comparison.json``."""
    destination = Path(path) if path is not None else DEFAULT_MODEL_COMPARISON_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(build_model_comparison(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def _dataset_comparison(dataset: Any, runs: list[dict[str, Any]]) -> dict[str, Any]:
    experiment = runs[0].get("experiment") if isinstance(runs[0].get("experiment"), dict) else {}
    models = []
    for run in runs:
        info = run.get("experiment") if isinstance(run.get("experiment"), dict) else {}
        model = info.get("model")
        if model not in models:
            models.append(model)
    baseline = [_baseline_f1(run) for run in runs]
    slices = [_worst_slice(run) for run in runs]
    errors = [_high_confidence_errors(run) for run in runs]
    uncertainty = [_mean_uncertainty(run) for run in runs]
    robustness = [_robustness(run) for run in runs]
    importance = [_importance(run) for run in runs]
    return {
        "dataset": dataset,
        "models": models,
        "random_state": experiment.get("random_state"),
        "test_size": experiment.get("test_size"),
        "metrics": [
            _metric("baseline_f1", "Overall F1", HIGHER_IS_BETTER, baseline),
            _metric(
                "worst_slice_delta_f1",
                "Worst failure-slice F1 delta",
                HIGHER_IS_BETTER,
                slices,
                note=(
                    "The value is the stored slice F1 minus the stored overall F1. "
                    "A more negative delta is greater degradation."
                ),
            ),
            _metric(
                "high_confidence_errors",
                "High-confidence errors",
                LOWER_IS_BETTER,
                errors,
                note=(
                    "The value is the stored count of incorrect predictions "
                    "at or above the confidence threshold."
                ),
            ),
            _metric(
                "mean_uncertainty",
                "Mean uncertainty",
                LOWER_IS_BETTER,
                uncertainty,
                note="The value is the stored mean uncertainty across the test rows.",
            ),
            _robustness_metric(robustness),
            _importance_metric(importance),
        ],
        "findings": {
            "strongest_baseline_model": _extreme(
                baseline,
                higher_is_better=True,
                selection="highest stored baseline F1",
            ),
            "most_robust_model": _most_robust(robustness),
            "greatest_failure_slice_degradation": _extreme(
                slices,
                higher_is_better=False,
                selection="lowest stored worst-slice F1 delta",
            ),
            "most_high_confidence_errors": _extreme(
                errors,
                higher_is_better=True,
                selection="highest stored high-confidence error count",
            ),
        },
    }


def _baseline_f1(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("baseline_metrics")
    value = None
    if isinstance(metrics, dict):
        value = _number(metrics.get("f1"))
    if value is None:
        return _unavailable(run, "insufficient_data", "Baseline F1 was not stored.")
    return _available(run, value)


def _worst_slice(run: dict[str, Any]) -> dict[str, Any]:
    component = _component(run, "slicing")
    if component.get("status") != "supported":
        return _from_component(run, component, "No stored failure slice is available.")
    stored = component.get("result") if isinstance(component.get("result"), dict) else {}
    worst = [item for item in stored.get("worst_slices") or [] if isinstance(item, dict)]
    measured = [item for item in worst if _number(item.get("delta_f1")) is not None]
    if not measured:
        return _unavailable(run, "insufficient_data", "No valid failure slice was stored.")
    chosen = min(measured, key=_slice_key)
    row = _available(run, _number(chosen.get("delta_f1")))
    row["feature"] = chosen.get("feature")
    row["slice"] = chosen.get("slice_definition")
    row["sample_count"] = chosen.get("sample_count")
    return row


def _high_confidence_errors(run: dict[str, Any]) -> dict[str, Any]:
    component = _component(run, "hard_examples")
    if component.get("status") != "supported":
        return _from_component(run, component, "High-confidence error counts were not stored.")
    stored = component.get("result") if isinstance(component.get("result"), dict) else {}
    summary = stored.get("summary") if isinstance(stored.get("summary"), dict) else {}
    count = _count(summary.get("wrong_confident"))
    if count is None:
        return _unavailable(
            run,
            "insufficient_data",
            "The stored high-confidence error count was not available.",
        )
    return _available(run, count)


def _mean_uncertainty(run: dict[str, Any]) -> dict[str, Any]:
    component = _component(run, "hard_examples")
    if component.get("status") != "supported":
        return _from_component(run, component, "Mean uncertainty was not stored.")
    stored = component.get("result") if isinstance(component.get("result"), dict) else {}
    summary = stored.get("summary") if isinstance(stored.get("summary"), dict) else {}
    value = _number(summary.get("mean_uncertainty"))
    if value is None:
        return _unavailable(
            run,
            "insufficient_data",
            "Mean uncertainty was not stored for the full test set.",
        )
    return _available(run, value)


def _robustness(run: dict[str, Any]) -> dict[str, Any]:
    component = _component(run, "robustness")
    model = _model_name(run)
    if component.get("status") != "supported":
        return {
            "model": model,
            "status": component.get("status"),
            "detail": component.get("detail") or "Robustness drops were not stored.",
            "levels": [],
        }
    stored = component.get("result") if isinstance(component.get("result"), dict) else {}
    levels = []
    for level in stored.get("results") or []:
        if not isinstance(level, dict):
            continue
        noise_level = _number(level.get("noise_level"))
        degradation = level.get("degradation") if isinstance(level.get("degradation"), dict) else {}
        drop = _number(degradation.get("f1"))
        if noise_level is None or drop is None:
            continue
        mean = level.get("mean") if isinstance(level.get("mean"), dict) else {}
        std = level.get("std") if isinstance(level.get("std"), dict) else {}
        levels.append(
            {
                "noise_level": noise_level,
                "f1_drop": drop,
                "f1_mean": _number(mean.get("f1")),
                "f1_std": _number(std.get("f1")),
            }
        )
    if not levels:
        return {
            "model": model,
            "status": "insufficient_data",
            "detail": "No stored robustness F1 drop was available.",
            "levels": [],
        }
    return {"model": model, "status": "supported", "detail": None, "levels": levels}


def _importance(run: dict[str, Any]) -> dict[str, Any]:
    component = _component(run, "explanation")
    model = _model_name(run)
    status = component.get("status")
    stored = component.get("result") if isinstance(component.get("result"), dict) else {}
    if status != "supported":
        return {
            "model": model,
            "status": status,
            "detail": _explanation_detail(component, stored),
            "features": [],
        }
    features = []
    for item in stored.get("features") or []:
        if not isinstance(item, dict):
            continue
        importance = _number(item.get("importance"))
        if importance is None or not item.get("feature"):
            continue
        features.append(
            {
                "feature": item.get("feature"),
                "importance": importance,
                "rank": _count(item.get("rank")),
            }
        )
    if not features:
        return {
            "model": model,
            "status": "insufficient_data",
            "detail": "Feature importance was not stored.",
            "features": [],
        }
    return {"model": model, "status": "supported", "detail": None, "features": features}


def _metric(name: str, label: str, direction: str, rows: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "label": label,
        "direction": direction,
        "models": rows,
    }
    if note is not None:
        payload["note"] = note
    return payload


def _robustness_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "robustness_f1_drop",
        "label": "Robustness F1 drop",
        "direction": LOWER_IS_BETTER,
        "note": "Each value is the stored baseline F1 minus the stored mean F1 at that noise level.",
        "models": rows,
    }


def _importance_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "feature_importance",
        "label": "Feature importance",
        "direction": HIGHER_IS_MORE_IMPORTANT_WITHIN_MODEL,
        "comparable_across_models": False,
        "note": (
            "Importance ranks features within one supported model. "
            "It is not a model-quality score and is not compared across models."
        ),
        "models": rows,
    }


def _extreme(rows: list[dict[str, Any]], *, higher_is_better: bool, selection: str) -> dict[str, Any]:
    available = [row for row in rows if row.get("status") == "supported" and _number(row.get("value")) is not None]
    if len(available) < _MIN_MODELS:
        return _not_identified(_TOO_FEW)
    best = max(float(row["value"]) for row in available) if higher_is_better else min(
        float(row["value"]) for row in available
    )
    tied = [row["model"] for row in available if float(row["value"]) == best]
    return {
        "identified": True,
        "models": tied,
        "value": _same_number(available, best),
        "tie": len(tied) > 1,
        "selection": selection,
        "detail": None,
    }


def _most_robust(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    for row in rows:
        if row.get("status") != "supported":
            continue
        levels = {}
        for level in row.get("levels") or []:
            noise_level = _number(level.get("noise_level"))
            drop = _number(level.get("f1_drop"))
            if noise_level is None or drop is None or noise_level in levels:
                continue
            levels[noise_level] = drop
        if levels:
            usable.append((row["model"], levels))
    if len(usable) < _MIN_MODELS:
        return _not_identified(_TOO_FEW)
    shared = set(usable[0][1])
    for _model, levels in usable[1:]:
        shared &= set(levels)
    if not shared:
        return _not_identified(
            "The models do not share a stored noise level, so the most robust model is not identified."
        )
    ordered_levels = sorted(shared, reverse=True)
    ranking = []
    for model, levels in usable:
        ranking.append((tuple(levels[level] for level in ordered_levels), model, levels))
    best = min(item[0] for item in ranking)
    tied = [model for drops, model, _levels in ranking if drops == best]
    best_levels = next(levels for drops, model, levels in ranking if model == tied[0])
    noise_level = ordered_levels[0]
    return {
        "identified": True,
        "models": tied,
        "value": best_levels[noise_level],
        "noise_level": noise_level,
        "tie": len(tied) > 1,
        "selection": (
            "Smallest stored F1 drop at the highest noise level shared by the compared models. "
            "Equal drops are then compared at the next shared noise level."
        ),
        "detail": None,
    }


def _not_identified(detail: str) -> dict[str, Any]:
    return {
        "identified": False,
        "models": [],
        "value": None,
        "tie": False,
        "selection": None,
        "detail": detail,
    }


def _available(run: dict[str, Any], value: int | float) -> dict[str, Any]:
    return {"model": _model_name(run), "status": "supported", "detail": None, "value": value}


def _unavailable(run: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
    return {"model": _model_name(run), "status": status, "detail": detail, "value": None}


def _from_component(run: dict[str, Any], component: dict[str, Any], fallback: str) -> dict[str, Any]:
    detail = component.get("detail") if isinstance(component.get("detail"), str) else None
    return _unavailable(run, str(component.get("status")), detail or fallback)


def _explanation_detail(component: dict[str, Any], stored: dict[str, Any]) -> str:
    reason = stored.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    detail = component.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return "Feature importance was not stored."


def _slice_key(item: dict[str, Any]) -> tuple:
    """Match the slicing rank: F1 delta, then accuracy delta, then names."""
    accuracy = _number(item.get("delta_accuracy"))
    return (
        float(item["delta_f1"]),
        accuracy is None,
        0.0 if accuracy is None else float(accuracy),
        str(item.get("feature")),
        str(item.get("slice_definition")),
    )


def _same_number(rows: list[dict[str, Any]], value: float) -> int | float:
    for row in rows:
        stored = row.get("value")
        if _number(stored) == value and isinstance(stored, (int, np.integer)) and not isinstance(stored, bool):
            return int(stored)
    return value


def _component(run: dict[str, Any], name: str) -> dict[str, Any]:
    value = run.get(name)
    if isinstance(value, dict):
        return value
    return {
        "status": "insufficient_data",
        "detail": "This component was not included in the diagnostic result.",
    }


def _model_name(run: dict[str, Any]) -> Any:
    experiment = run.get("experiment")
    if isinstance(experiment, dict):
        return experiment.get("model")
    return None


def _payload(result: DiagnosticEngineResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, DiagnosticEngineResult):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError("Model comparison expects a DiagnosticEngineResult or its dictionary.")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    return None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)
