"""Dataset and model names accepted by an experiment.

The names come from the Phase 1 loader and model zoo. This module does not
keep a second list.
"""

from __future__ import annotations

from src.data.data_loader import SUPPORTED_DATASETS
from src.models.models import MODEL_NAMES


def list_datasets() -> tuple[str, ...]:
    """Return the dataset names ``load_dataset`` accepts."""
    return SUPPORTED_DATASETS


def validate_dataset(name: str) -> str:
    """Return ``name`` when it is a supported dataset."""
    if not isinstance(name, str):
        raise TypeError("dataset must be a string.")
    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {supported}.")
    return name


def list_models() -> tuple[str, ...]:
    """Return the model names ``get_models`` provides."""
    return MODEL_NAMES


def validate_models(names: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return the selected model names in the order given.

    A single model name is accepted. An empty selection, a duplicate, or a
    name outside ``MODEL_NAMES`` is rejected.
    """
    if isinstance(names, str):
        selected = (names,)
    elif isinstance(names, (list, tuple)):
        selected = tuple(names)
    else:
        raise TypeError("models must be a model name or a list of model names.")
    if not selected:
        raise ValueError(_no_models_message())
    if any(not isinstance(name, str) for name in selected):
        raise TypeError("each model name must be a string.")
    if len(selected) != len(set(selected)):
        raise ValueError("models must not contain duplicate names.")
    unknown = [name for name in selected if name not in MODEL_NAMES]
    if unknown:
        supported = ", ".join(MODEL_NAMES)
        joined = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Unknown model(s) {joined}. Available models: {supported}.")
    return selected


def resolve_models(names: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Resolve a CLI or config selection, including the token ``all``.

    ``all`` is replaced by ``list_models()``. It is rejected when combined
    with individual model names.
    """
    if isinstance(names, str):
        selected = (names,)
    elif isinstance(names, (list, tuple)):
        selected = tuple(names)
    else:
        raise TypeError("models must be a model name or a list of model names.")
    if not selected:
        raise ValueError(_no_models_message())
    if "all" in selected:
        if selected != ("all",):
            available = ", ".join(list_models())
            raise ValueError(
                "Select 'all' by itself, or name specific models. "
                f"Available models: {available}."
            )
        return list_models()
    return validate_models(selected)


def _no_models_message() -> str:
    available = ", ".join(MODEL_NAMES)
    return f"No models were selected. Available models: {available}."
