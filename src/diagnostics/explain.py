"""Built-in feature importance for fitted tree models.

Importance comes from ``feature_importances_`` on the trained classifier.
Feature names come from the fitted preprocessing step's
``get_feature_names_out``. Those are the columns the classifier received.
This module does not fit a model and does not rename encoded columns.

Logistic regression is not given an importance score. This project has no
coefficient explanation. A model without ``feature_importances_`` returns a
``not_supported`` result.

SHAP is optional future work. It is not a dependency and is not imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Ranked built-in importances, or a not-supported result."""

    dataset: str | None
    model: str | None
    status: str
    method: str | None
    reason: str | None
    features: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready explanation for one model."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "status": self.status,
            "method": self.method,
            "reason": self.reason,
            "features": [dict(row) for row in self.features],
        }


def explain_feature_importance(
    model,
    *,
    dataset: str | None = None,
    model_name: str | None = None,
) -> FeatureImportanceResult:
    """Rank ``feature_importances_`` using the preprocessing output names.

    Models without that attribute return ``status="not_supported"`` and an
    empty feature list. They do not raise.
    """
    estimator, preprocessor = _split_model(model)
    importances = _importances(estimator)
    if importances is None:
        return _not_supported(
            dataset,
            model_name,
            "This model does not expose feature_importances_.",
        )
    names = _transformed_names(preprocessor, importances.shape[0])
    if names is None:
        return _not_supported(
            dataset,
            model_name,
            "Feature names could not be recovered from the preprocessing pipeline.",
        )
    return FeatureImportanceResult(
        dataset=dataset,
        model=model_name,
        status="ok",
        method="feature_importances_",
        reason=None,
        features=tuple(_ranked_rows(names, importances)),
    )


def save_feature_importance(result: FeatureImportanceResult, directory: Path | str) -> Path:
    """Write ``<dataset>_<model>_feature_importance.json`` under ``directory``."""
    if not result.dataset or not result.model:
        raise ValueError("Saving feature importance requires both a dataset name and a model name.")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.dataset}_{result.model}_feature_importance.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def explanations_directory(model_dir: Path | str) -> Path:
    """Return ``outputs/diagnostics/explanations`` for a normal experiment directory."""
    directory = Path(model_dir)
    if directory.name == "experiments":
        return directory.parent / "diagnostics" / "explanations"
    if directory.parent.name == "experiments":
        return directory.parent.parent / "diagnostics" / "explanations"
    return directory / "diagnostics" / "explanations"


def _split_model(model):
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        estimator = steps.get("classifier")
        if estimator is None and getattr(model, "steps", None):
            estimator = model.steps[-1][1]
        return estimator, steps.get("preprocessing")
    return model, None


def _importances(estimator):
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None
    try:
        values = np.asarray(estimator.feature_importances_, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def _transformed_names(preprocessor, count: int) -> list[str] | None:
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return None
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, NotImplementedError, ValueError, TypeError):
        return None
    if len(names) != count or len(set(names)) != len(names):
        return None
    return names


def _ranked_rows(names: list[str], importances) -> list[dict[str, Any]]:
    order = sorted(
        range(len(names)),
        key=lambda index: (-float(importances[index]), names[index]),
    )
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "feature": names[index],
                "importance": float(importances[index]),
                "rank": rank,
            }
        )
    return rows


def _not_supported(dataset, model_name, reason: str) -> FeatureImportanceResult:
    return FeatureImportanceResult(
        dataset=dataset,
        model=model_name,
        status="not_supported",
        method=None,
        reason=reason,
        features=(),
    )
