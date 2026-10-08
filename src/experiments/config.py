"""Load and validate the experiment configuration.

JSON is read only in this module. Dataset and model names are checked with
the registry before a configuration object is returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.experiments.registry import resolve_models, validate_dataset, validate_models

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.json"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

_FIELDS = {"dataset", "models", "random_state", "test_size", "output_dir"}
_REQUIRED = ("dataset", "models", "output_dir")
_PHASE1_OUTPUT_DIRS = (
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "outputs" / "metrics",
    _PROJECT_ROOT / "outputs" / "errors",
    _PROJECT_ROOT / "outputs" / "reports",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings for one experiment.

    ``random_state`` is 42 and ``test_size`` is 0.2 unless the configuration
    sets another value. Model hyperparameters are not part of this object.
    """

    dataset: str
    models: tuple[str, ...]
    random_state: int
    test_size: float
    output_dir: Path


def apply_overrides(
    config: ExperimentConfig,
    *,
    dataset: str | None = None,
    models: str | list[str] | tuple[str, ...] | None = None,
    random_state: int | None = None,
    test_size: float | None = None,
) -> ExperimentConfig:
    """Return a new config with command-line values applied.

    Omitted overrides keep the loaded configuration. Dataset names, model
    names, ``random_state``, and ``test_size`` are validated again here.
    """
    selected_dataset = config.dataset if dataset is None else validate_dataset(dataset)
    selected_models = config.models if models is None else resolve_models(models)
    selected_seed = (
        config.random_state
        if random_state is None
        else _require_int(random_state, "random_state")
    )
    selected_size = config.test_size if test_size is None else _require_test_size(test_size)
    return ExperimentConfig(
        dataset=selected_dataset,
        models=selected_models,
        random_state=selected_seed,
        test_size=selected_size,
        output_dir=config.output_dir,
    )


def load_config(path: Path | str | None = None) -> ExperimentConfig:
    """Read a JSON configuration file and return a validated config.

    Parameters
    ----------
    path:
        Configuration file. Defaults to the project ``config.json``.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a JSON object.")
    return _from_mapping(payload)


def _from_mapping(payload: dict) -> ExperimentConfig:
    unknown = sorted(set(payload) - _FIELDS)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(
            f"Unknown configuration field(s): {joined}. "
            "Hyperparameters are not configurable."
        )
    missing = [field for field in _REQUIRED if field not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Configuration is missing required field(s): {joined}.")

    dataset = validate_dataset(payload["dataset"])
    models = validate_models(payload["models"])
    random_state = _require_int(
        payload.get("random_state", DEFAULT_RANDOM_STATE),
        "random_state",
    )
    test_size = _require_test_size(payload.get("test_size", DEFAULT_TEST_SIZE))
    output_dir = _resolve_output_dir(payload["output_dir"])
    return ExperimentConfig(
        dataset=dataset,
        models=models,
        random_state=random_state,
        test_size=test_size,
        output_dir=output_dir,
    )


def _require_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return value


def _require_test_size(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("test_size must be a number between 0 and 1.")
    test_size = float(value)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    return test_size


def _resolve_output_dir(value) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_dir must be a non-empty path.")
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    resolved = path.resolve()
    for blocked in _PHASE1_OUTPUT_DIRS:
        blocked_resolved = blocked.resolve()
        if resolved == blocked_resolved or blocked_resolved in resolved.parents:
            raise ValueError(
                "output_dir must stay outside Phase 1 model, metric, error, and report outputs. "
                f"Got {value!r}."
            )
    return resolved
