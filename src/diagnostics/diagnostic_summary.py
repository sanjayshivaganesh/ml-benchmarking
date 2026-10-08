"""Join feature importance with failure slices.

A joined result is an association: a raw feature ranks high in built-in
importance, and one of its reliable slices has lower F1 than the full test
set. The summary does not say that the feature caused the errors.

Transformed names from preprocessing are mapped back to sliced raw features
only when the name is the raw feature or starts with that feature name plus
an underscore, as one-hot columns do. The longest matching raw name wins.
Grouped importance is the sum of those transformed importances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.diagnostics.explain import FeatureImportanceResult
from src.diagnostics.slicing import ErrorSlicingResult

DEFAULT_TOP_FEATURES = 5
DEFAULT_SUBSTANTIAL_F1_DROP = 0.05
_NOTE = (
    "These statements describe associations between built-in feature importance "
    "and slice performance. They do not show that a feature caused the model to fail."
)


@dataclass(frozen=True)
class DiagnosticSummary:
    """Deterministic importance-and-slice associations for one model."""

    dataset: str | None
    model: str | None
    importance_status: str
    overall_f1: float | None
    top_feature_count: int
    substantial_f1_drop: float
    associations: tuple[dict[str, Any], ...]
    table: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary for a later report."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "association_only": True,
            "note": _NOTE,
            "importance_status": self.importance_status,
            "overall_f1": self.overall_f1,
            "top_feature_count": self.top_feature_count,
            "substantial_f1_drop": self.substantial_f1_drop,
            "associations": [dict(item) for item in self.associations],
        }


def build_diagnostic_summary(
    importance: FeatureImportanceResult,
    slicing: ErrorSlicingResult,
    *,
    top_features: int = DEFAULT_TOP_FEATURES,
    substantial_f1_drop: float = DEFAULT_SUBSTANTIAL_F1_DROP,
) -> DiagnosticSummary:
    """Associate high-importance raw features with their worst reliable slice.

    A feature is included when its grouped importance rank is within
    ``top_features`` and its worst reliable slice has a negative F1 delta.
    The delta is substantial when it is at or below ``-substantial_f1_drop``.
    """
    if not isinstance(importance, FeatureImportanceResult):
        raise TypeError("importance must be a FeatureImportanceResult.")
    if not isinstance(slicing, ErrorSlicingResult):
        raise TypeError("slicing must be an ErrorSlicingResult.")
    _require_same_result(importance, slicing)
    limit = _require_positive_int(top_features, "top_features")
    drop = _require_nonnegative_float(substantial_f1_drop, "substantial_f1_drop")
    dataset = importance.dataset or slicing.dataset
    model = importance.model or slicing.model
    overall_f1 = _overall_f1(slicing)
    if importance.status != "ok":
        associations: tuple[dict[str, Any], ...] = ()
    else:
        associations = tuple(
            _associations(importance, slicing, limit, drop, overall_f1)
        )
    return DiagnosticSummary(
        dataset=dataset,
        model=model,
        importance_status=importance.status,
        overall_f1=overall_f1,
        top_feature_count=limit,
        substantial_f1_drop=drop,
        associations=associations,
        table=_association_table(associations),
    )


def save_diagnostic_summary(result: DiagnosticSummary, directory: Path | str) -> Path:
    """Write ``<dataset>_<model>_diagnostic_summary.json`` under ``directory``."""
    if not result.dataset or not result.model:
        raise ValueError("Saving a diagnostic summary requires both a dataset name and a model name.")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.dataset}_{result.model}_diagnostic_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def summary_directory(model_dir: Path | str) -> Path:
    """Return ``outputs/diagnostics/summaries`` for a normal experiment directory."""
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "summaries"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "summaries"
    return directory / "diagnostics" / "summaries"


def _associations(importance, slicing, limit: int, drop: float, overall_f1: float | None):
    raw_features = sorted({str(item["feature"]) for item in slicing.slices})
    grouped = _group_importances(importance.features, raw_features)
    ranked = _rank_groups(grouped)
    worst = _worst_by_feature(slicing.slices)
    rows = []
    for group in ranked:
        if group["importance_rank"] > limit:
            continue
        slice_row = worst.get(group["feature"])
        if slice_row is None:
            continue
        delta = slice_row["delta_f1"]
        if delta is None or delta >= 0:
            continue
        substantial = delta <= -drop
        statement = _statement(group["feature"], slice_row["slice_definition"], substantial)
        rows.append(
            {
                "feature": group["feature"],
                "importance": group["importance"],
                "importance_rank": group["importance_rank"],
                "transformed_features": group["transformed_features"],
                "worst_slice": slice_row["slice_definition"],
                "f1": slice_row["metrics"]["f1"],
                "overall_f1": overall_f1,
                "delta_f1": delta,
                "substantial": substantial,
                "statement": statement,
            }
        )
    return rows


def _group_importances(features, raw_features: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in features:
        raw_name = _raw_feature(str(row["feature"]), raw_features)
        if raw_name is None:
            continue
        group = grouped.setdefault(
            raw_name,
            {"feature": raw_name, "importance": 0.0, "transformed_features": []},
        )
        group["importance"] = float(group["importance"]) + float(row["importance"])
        group["transformed_features"].append(
            {"feature": str(row["feature"]), "importance": float(row["importance"])}
        )
    for group in grouped.values():
        group["transformed_features"] = tuple(
            sorted(group["transformed_features"], key=lambda item: item["feature"])
        )
    return grouped


def _raw_feature(transformed: str, raw_features: list[str]) -> str | None:
    if transformed in raw_features:
        return transformed
    matches = [
        name for name in raw_features if transformed.startswith(name + "_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _rank_groups(grouped: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        grouped.values(),
        key=lambda item: (-float(item["importance"]), item["feature"]),
    )
    ranked = []
    for rank, item in enumerate(ordered, start=1):
        ranked.append({**item, "importance_rank": rank})
    return ranked


def _worst_by_feature(slices) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for item in slices:
        if not item.get("reliable") or item.get("delta_f1") is None:
            continue
        current = chosen.get(item["feature"])
        if current is None or _slice_sort_key(item) < _slice_sort_key(current):
            chosen[str(item["feature"])] = item
    return chosen


def _slice_sort_key(item: dict[str, Any]):
    return (
        item["delta_f1"],
        item.get("delta_accuracy"),
        item["slice_definition"],
    )


def _statement(feature: str, definition: str, substantial: bool) -> str:
    if substantial:
        observed = "exhibits substantial performance degradation"
    else:
        observed = "has lower F1 than the full test set"
    return (
        f"{feature} is among the model's most important features, "
        f"while the slice {definition} {observed}."
    )


def _association_table(associations) -> pd.DataFrame:
    columns = (
        "feature",
        "importance",
        "importance_rank",
        "worst_slice",
        "f1",
        "overall_f1",
        "delta_f1",
        "substantial",
        "statement",
    )
    if not associations:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([{column: item[column] for column in columns} for item in associations])


def _overall_f1(slicing: ErrorSlicingResult) -> float | None:
    metrics = slicing.overall_metrics or {}
    if "f1" not in metrics:
        return None
    return float(metrics["f1"])


def _require_same_result(importance: FeatureImportanceResult, slicing: ErrorSlicingResult) -> None:
    if (
        importance.dataset
        and slicing.dataset
        and importance.dataset != slicing.dataset
    ):
        raise ValueError(
            "Feature importance and slicing refer to different datasets: "
            f"{importance.dataset!r} and {slicing.dataset!r}."
        )
    if importance.model and slicing.model and importance.model != slicing.model:
        raise ValueError(
            "Feature importance and slicing refer to different models: "
            f"{importance.model!r} and {slicing.model!r}."
        )


def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _require_nonnegative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be at least 0.")
    return number
