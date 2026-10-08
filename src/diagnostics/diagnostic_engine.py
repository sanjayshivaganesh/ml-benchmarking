"""Orchestrate Phase 3 diagnostics for configured experiments.

The dataset, models, seed, and output directory come from an
``ExperimentConfig``. Saved predictions and model files are loaded. The
experiment is trained only when those artifacts are missing. A model without
feature importance does not stop the other diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd

from src.data.data_loader import load_dataset
from src.diagnostics.diagnostic_summary import (
    build_diagnostic_summary,
    save_diagnostic_summary,
    summary_directory,
)
from src.diagnostics.explain import (
    FeatureImportanceResult,
    explain_feature_importance,
    explanations_directory,
    save_feature_importance,
)
from src.diagnostics.hard_examples import (
    explore_hard_examples,
    hard_examples_directory,
    save_hard_examples,
)
from src.diagnostics.prediction_store import PredictionStore
from src.diagnostics.robustness import (
    DEFAULT_N_SEEDS,
    DEFAULT_NOISE_LEVELS,
    robustness_directory,
    run_robustness,
    save_robustness_result,
)
from src.diagnostics.slicing import (
    ErrorSlicingResult,
    run_error_slicing,
    save_slicing_result,
    slicing_directory,
)
from src.diagnostics.workflow import _has_saved_models, _require_same_experiment
from src.experiments.config import ExperimentConfig
from src.experiments.registry import validate_dataset
from src.run_phase2 import (
    STEP_HARD_EXAMPLES,
    STEP_ROBUSTNESS,
    STEP_SLICING,
    make_experiment_id,
    run_phase2,
)

STATUS_SUPPORTED = "supported"
STATUS_NOT_SUPPORTED = "not_supported"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_FAILED = "failed"
_COMPONENT_ERRORS = (ValueError, TypeError, FileNotFoundError, OSError)
_DISABLED_DETAIL = "Disabled in the experiment configuration."


@dataclass(frozen=True)
class DiagnosticComponent:
    """One diagnostic step and its explicit status."""

    name: str
    status: str
    detail: str | None = None
    output_path: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "output_path": self.output_path,
            "result": self.result,
        }


@dataclass(frozen=True)
class ModelDiagnostic:
    """Diagnostics for one dataset and one model."""

    experiment: dict[str, Any]
    baseline_metrics: dict[str, float] | None
    prediction_validation: DiagnosticComponent
    slicing: DiagnosticComponent
    hard_examples: DiagnosticComponent
    robustness: DiagnosticComponent
    explanation: DiagnosticComponent
    diagnostic_summary: DiagnosticComponent

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": dict(self.experiment),
            "baseline_metrics": (
                None if self.baseline_metrics is None else dict(self.baseline_metrics)
            ),
            "prediction_validation": self.prediction_validation.to_dict(),
            "slicing": self.slicing.to_dict(),
            "hard_examples": self.hard_examples.to_dict(),
            "robustness": self.robustness.to_dict(),
            "explanation": self.explanation.to_dict(),
            "diagnostic_summary": self.diagnostic_summary.to_dict(),
        }


@dataclass(frozen=True)
class DiagnosticEngineResult:
    """Structured diagnostics for every selected dataset and model."""

    datasets: tuple[str, ...]
    models: tuple[str, ...]
    random_state: int
    test_size: float
    output_dir: Path
    runs: tuple[ModelDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": list(self.datasets),
            "models": list(self.models),
            "random_state": self.random_state,
            "test_size": self.test_size,
            "output_dir": str(self.output_dir),
            "runs": [run.to_dict() for run in self.runs],
        }


def run_diagnostics(
    config: ExperimentConfig,
    *,
    datasets: Sequence[str] | None = None,
    noise_levels: Sequence[float] | None = None,
    n_seeds: int | None = None,
    features: Sequence[str] | None = None,
    slicing: bool = True,
    hard_examples: bool = True,
    robustness: bool = True,
    explanations: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DiagnosticEngineResult:
    """Run slicing, hard examples, robustness, explanations, and cross-analysis.

    ``datasets`` overrides ``config.dataset`` when several datasets should
    share the same models, seed, and test size. Names are checked with the
    registry. Each dataset uses its own saved experiment. A false component
    flag skips that step. ``progress`` is called once before each enabled
    stage, after saved artifacts have been loaded or trained.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("run_diagnostics expects an ExperimentConfig.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable.")
    selected = _selected_datasets(config, datasets)
    levels = DEFAULT_NOISE_LEVELS if noise_levels is None else noise_levels
    seeds = DEFAULT_N_SEEDS if n_seeds is None else n_seeds
    slicing = _require_flag(slicing, "slicing")
    hard_examples = _require_flag(hard_examples, "hard_examples")
    robustness = _require_flag(robustness, "robustness")
    explanations = _require_flag(explanations, "explanations")
    bundles = [
        _prepare_dataset(
            replace(config, dataset=dataset),
            noise_levels=levels,
            n_seeds=seeds,
            features=features,
        )
        for dataset in selected
    ]
    _run_stage(bundles, "slicing", slicing, STEP_SLICING, progress, blocked=True)
    _run_stage(bundles, "hard_examples", hard_examples, STEP_HARD_EXAMPLES, progress, blocked=True)
    _run_stage(bundles, "robustness", robustness, STEP_ROBUSTNESS, progress, blocked=True)
    _run_stage(bundles, "explanation", explanations, None, progress, blocked=False)
    runs: list[ModelDiagnostic] = []
    for bundle in bundles:
        for context in bundle["contexts"]:
            summary = _run_summary(
                bundle["experiment_dir"],
                context["slicing"],
                context["explanation"],
            )
            runs.append(
                ModelDiagnostic(
                    experiment=context["experiment"],
                    baseline_metrics=context["baseline"],
                    prediction_validation=context["validation"],
                    slicing=context["slicing"],
                    hard_examples=context["hard_examples"],
                    robustness=context["robustness"],
                    explanation=context["explanation"],
                    diagnostic_summary=summary,
                )
            )
    return DiagnosticEngineResult(
        datasets=selected,
        models=tuple(config.models),
        random_state=config.random_state,
        test_size=config.test_size,
        output_dir=config.output_dir,
        runs=tuple(runs),
    )


def format_diagnostic_engine(result: DiagnosticEngineResult) -> str:
    """Return a short status list for the diagnostic run."""
    lines = [
        f"Datasets: {', '.join(result.datasets)}",
        f"Models: {', '.join(result.models)}",
        f"random_state: {result.random_state}",
        f"test_size: {result.test_size}",
    ]
    for run in result.runs:
        experiment = run.experiment
        lines.append(f"{experiment['dataset']} / {experiment['model']}")
        for component in (
            run.prediction_validation,
            run.slicing,
            run.hard_examples,
            run.robustness,
            run.explanation,
            run.diagnostic_summary,
        ):
            lines.append(f"  {component.name}: {component.status}")
    return "\n".join(lines)


def _prepare_dataset(
    config: ExperimentConfig,
    *,
    noise_levels: Sequence[float],
    n_seeds: int,
    features: Sequence[str] | None,
) -> dict[str, Any]:
    experiment_id = make_experiment_id(
        config.dataset,
        config.models,
        config.random_state,
        config.test_size,
    )
    experiment_dir = config.output_dir / experiment_id
    reused = _has_saved_models(experiment_dir, config.models)
    if not reused:
        run_phase2(config)
    prepared, problem = _load_experiment(config, experiment_dir)
    contexts = []
    for model_name in config.models:
        metadata = {} if prepared is None else prepared["metadata"]
        validation = _validate_model(prepared, problem, model_name)
        contexts.append(
            {
                "model_name": model_name,
                "experiment": _experiment_record(
                    config,
                    model_name,
                    experiment_id,
                    experiment_dir,
                    reused,
                    metadata,
                ),
                "baseline": _baseline_metrics(prepared, model_name),
                "validation": validation,
                "blocked": _blocked(validation),
                "slicing": None,
                "hard_examples": None,
                "robustness": None,
                "explanation": None,
            }
        )
    return {
        "config": config,
        "experiment_dir": experiment_dir,
        "prepared": prepared,
        "contexts": contexts,
        "noise_levels": noise_levels,
        "n_seeds": n_seeds,
        "features": features,
    }


def _run_stage(bundles, name: str, enabled: bool, label: str | None, progress, *, blocked: bool) -> None:
    if enabled and label is not None:
        _emit(progress, label)
    for bundle in bundles:
        for context in bundle["contexts"]:
            if not enabled:
                context[name] = _disabled_component(name)
                continue
            context[name] = _component(
                name,
                context["blocked"] if blocked else None,
                lambda context=context, bundle=bundle, name=name: _run_named_stage(
                    name,
                    bundle,
                    context,
                ),
            )


def _run_named_stage(name: str, bundle: dict[str, Any], context: dict[str, Any]) -> DiagnosticComponent:
    config = bundle["config"]
    model_name = context["model_name"]
    prepared = bundle["prepared"]
    experiment_dir = bundle["experiment_dir"]
    if name == "slicing":
        return _run_slicing(config, model_name, prepared, experiment_dir)
    if name == "hard_examples":
        return _run_hard_examples(config, model_name, prepared, experiment_dir)
    if name == "robustness":
        return _run_robustness(
            config,
            model_name,
            prepared,
            experiment_dir,
            noise_levels=bundle["noise_levels"],
            n_seeds=bundle["n_seeds"],
            features=bundle["features"],
        )
    if name == "explanation":
        return _run_explanation(config, model_name, prepared, experiment_dir)
    raise RuntimeError(f"Unknown diagnostic stage {name!r}.")


def _emit(progress, label: str) -> None:
    if progress is not None:
        progress(label)


def _disabled_component(name: str) -> DiagnosticComponent:
    return DiagnosticComponent(name, STATUS_NOT_SUPPORTED, _DISABLED_DETAIL)


def _require_flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _load_experiment(
    config: ExperimentConfig,
    experiment_dir: Path,
) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
    try:
        metadata = _read_json(experiment_dir / "metadata.json")
        predictions = _read_json(experiment_dir / "predictions.json")
        metrics = _read_json(experiment_dir / "metrics.json")
        _require_same_experiment(config, metadata)
    except _COMPONENT_ERRORS as exc:
        return None, (STATUS_FAILED, str(exc))
    prepared = {
        "metadata": metadata,
        "predictions": predictions,
        "metrics": metrics,
        "features": None,
        "labels": None,
    }
    try:
        sample_ids = predictions.get("test_index")
        if not isinstance(sample_ids, list) or not sample_ids:
            return prepared, (
                STATUS_INSUFFICIENT_DATA,
                "Saved predictions do not contain any test rows.",
            )
        split = load_dataset(
            config.dataset,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        features = split.X_test.loc[list(sample_ids)]
        if _identifiers(features.index) != _identifiers(sample_ids):
            raise ValueError(
                "Saved prediction sample ids do not match the reloaded test split."
            )
        saved_labels = np.asarray(predictions.get("y_test"))
        reloaded = np.asarray(split.y_test.loc[features.index].to_numpy())
        if reloaded.shape != saved_labels.shape or not np.array_equal(reloaded, saved_labels):
            raise ValueError("Saved labels do not match the reloaded test split.")
    except _COMPONENT_ERRORS as exc:
        return prepared, (STATUS_FAILED, str(exc))
    prepared["features"] = features
    prepared["labels"] = split.y_test.loc[features.index]
    return prepared, None


def _validate_model(
    prepared: dict[str, Any] | None,
    problem: tuple[str, str] | None,
    model_name: str,
) -> DiagnosticComponent:
    if problem is not None:
        status, detail = problem
        return DiagnosticComponent("prediction_validation", status, detail)
    try:
        predictions = prepared["predictions"]
        models = predictions.get("models")
        if not isinstance(models, dict) or model_name not in models:
            raise ValueError(f"Saved predictions have no rows for {model_name!r}.")
        entry = models[model_name]
        predicted = entry.get("predictions") if isinstance(entry, dict) else None
        if not isinstance(predicted, list) or not predicted:
            return DiagnosticComponent(
                "prediction_validation",
                STATUS_INSUFFICIENT_DATA,
                f"Saved predictions for {model_name!r} do not contain any test rows.",
            )
        store = PredictionStore.from_phase2_predictions(predictions, model_name)
        if len(store.frame) != len(prepared["labels"]):
            raise ValueError(
                f"Saved predictions for {model_name!r} do not match the test split."
            )
        saved = store.frame["y_true"].to_numpy()
        reloaded = np.asarray(prepared["labels"].to_numpy())
        if not np.array_equal(saved, reloaded):
            raise ValueError(
                f"Saved labels for {model_name!r} do not match the reloaded test split."
            )
    except _COMPONENT_ERRORS as exc:
        return DiagnosticComponent("prediction_validation", STATUS_FAILED, str(exc))
    return DiagnosticComponent("prediction_validation", STATUS_SUPPORTED)


def _run_slicing(config, model_name, prepared, experiment_dir) -> DiagnosticComponent:
    metadata = prepared["metadata"]
    store = PredictionStore.from_phase2_predictions(prepared["predictions"], model_name)
    slicing = run_error_slicing(
        prepared["features"],
        store.frame["y_true"].to_numpy(),
        store.frame["y_pred"].to_numpy(),
        dataset=config.dataset,
        model=model_name,
        numeric_features=metadata.get("numeric_features"),
        categorical_features=metadata.get("categorical_features"),
    )
    if not slicing.slices:
        return DiagnosticComponent(
            "slicing",
            STATUS_INSUFFICIENT_DATA,
            "Error slicing produced no slices.",
            result=slicing.to_dict(),
        )
    path = save_slicing_result(slicing, slicing_directory(experiment_dir))
    return _supported("slicing", path, slicing.to_dict())


def _run_hard_examples(config, model_name, prepared, experiment_dir) -> DiagnosticComponent:
    store = PredictionStore.from_phase2_predictions(
        prepared["predictions"],
        model_name,
        features=prepared["features"],
    )
    if store.frame.empty:
        return DiagnosticComponent(
            "hard_examples",
            STATUS_INSUFFICIENT_DATA,
            "Hard-example analysis has no test rows.",
        )
    explored = explore_hard_examples(
        store,
        dataset=config.dataset,
        model=model_name,
    )
    path = save_hard_examples(explored, hard_examples_directory(experiment_dir))
    return _supported("hard_examples", path, explored.to_dict())


def _run_robustness(
    config,
    model_name,
    prepared,
    experiment_dir,
    *,
    noise_levels,
    n_seeds,
    features,
) -> DiagnosticComponent:
    model = _load_model(prepared["metadata"], experiment_dir, model_name)
    metadata = prepared["metadata"]
    numeric = metadata.get("numeric_features") or []
    if features is not None and len(list(features)) == 0:
        return DiagnosticComponent(
            "robustness",
            STATUS_INSUFFICIENT_DATA,
            "No numerical features were selected for perturbation.",
        )
    if features is None and isinstance(numeric, list) and len(numeric) == 0:
        return DiagnosticComponent(
            "robustness",
            STATUS_INSUFFICIENT_DATA,
            "The dataset has no numerical features to perturb.",
        )
    result = run_robustness(
        model,
        prepared["features"],
        prepared["labels"],
        dataset=config.dataset,
        model_name=model_name,
        numeric_features=metadata.get("numeric_features"),
        categorical_features=metadata.get("categorical_features"),
        features=features,
        noise_levels=noise_levels,
        n_seeds=n_seeds,
        random_state=config.random_state,
    )
    perturbed = result.noise_configuration.get("perturbed_features") or ()
    path = save_robustness_result(result, robustness_directory(experiment_dir))
    if len(perturbed) == 0:
        return DiagnosticComponent(
            "robustness",
            STATUS_INSUFFICIENT_DATA,
            "No numerical features were perturbed.",
            output_path=str(path),
            result=result.to_dict(),
        )
    return _supported("robustness", path, result.to_dict())


def _run_explanation(config, model_name, prepared, experiment_dir) -> DiagnosticComponent:
    if prepared is None:
        raise FileNotFoundError(f"Saved experiment metadata is missing for {model_name!r}.")
    model = _load_model(prepared["metadata"], experiment_dir, model_name)
    explained = explain_feature_importance(
        model,
        dataset=config.dataset,
        model_name=model_name,
    )
    path = save_feature_importance(explained, explanations_directory(experiment_dir))
    payload = explained.to_dict()
    if explained.status == "not_supported":
        return DiagnosticComponent(
            "explanation",
            STATUS_NOT_SUPPORTED,
            explained.reason,
            output_path=str(path),
            result=payload,
        )
    if explained.status != "ok":
        return DiagnosticComponent(
            "explanation",
            STATUS_FAILED,
            explained.reason or "Feature importance did not complete.",
            output_path=str(path),
            result=payload,
        )
    return _supported("explanation", path, payload)


def _run_summary(experiment_dir, slicing, explanation) -> DiagnosticComponent:
    if slicing.detail == _DISABLED_DETAIL or explanation.detail == _DISABLED_DETAIL:
        return DiagnosticComponent("diagnostic_summary", STATUS_NOT_SUPPORTED, _DISABLED_DETAIL)
    if slicing.status == STATUS_INSUFFICIENT_DATA:
        return DiagnosticComponent(
            "diagnostic_summary",
            STATUS_INSUFFICIENT_DATA,
            slicing.detail,
        )
    if slicing.status != STATUS_SUPPORTED or slicing.result is None:
        return DiagnosticComponent(
            "diagnostic_summary",
            STATUS_FAILED,
            "Error slicing is unavailable, so importance and slices were not joined.",
        )
    if explanation.status == STATUS_FAILED or explanation.result is None:
        return DiagnosticComponent(
            "diagnostic_summary",
            STATUS_FAILED,
            "Feature importance is unavailable, so importance and slices were not joined.",
        )
    try:
        importance = _importance_from_payload(explanation.result)
        slices = _slicing_from_payload(slicing.result)
        summary = build_diagnostic_summary(importance, slices)
        path = save_diagnostic_summary(summary, summary_directory(experiment_dir))
    except _COMPONENT_ERRORS as exc:
        return DiagnosticComponent("diagnostic_summary", STATUS_FAILED, str(exc))
    payload = summary.to_dict()
    if explanation.status == STATUS_NOT_SUPPORTED:
        return DiagnosticComponent(
            "diagnostic_summary",
            STATUS_NOT_SUPPORTED,
            explanation.detail,
            output_path=str(path),
            result=payload,
        )
    return _supported("diagnostic_summary", path, payload)


def _component(
    name: str,
    blocked: DiagnosticComponent | None,
    func: Callable[[], DiagnosticComponent],
) -> DiagnosticComponent:
    if blocked is not None:
        return DiagnosticComponent(name, blocked.status, blocked.detail)
    try:
        return func()
    except _COMPONENT_ERRORS as exc:
        return DiagnosticComponent(name, STATUS_FAILED, str(exc))


def _blocked(validation: DiagnosticComponent) -> DiagnosticComponent | None:
    if validation.status == STATUS_SUPPORTED:
        return None
    return validation


def _supported(name: str, path: Path, payload: dict[str, Any]) -> DiagnosticComponent:
    return DiagnosticComponent(name, STATUS_SUPPORTED, output_path=str(path), result=payload)


def _load_model(metadata: dict, experiment_dir: Path, model_name: str):
    filenames = metadata.get("filenames")
    if not isinstance(filenames, dict) or not isinstance(filenames.get(model_name), str):
        raise FileNotFoundError(f"Saved experiment metadata has no file for {model_name!r}.")
    path = experiment_dir / filenames[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}.")
    return joblib.load(path)


def _importance_from_payload(payload: dict[str, Any]):
    return FeatureImportanceResult(
        dataset=payload.get("dataset"),
        model=payload.get("model"),
        status=str(payload.get("status")),
        method=payload.get("method"),
        reason=payload.get("reason"),
        features=tuple(payload.get("features") or ()),
    )


def _slicing_from_payload(payload: dict[str, Any]) -> ErrorSlicingResult:
    return ErrorSlicingResult(
        dataset=payload.get("dataset"),
        model=payload.get("model"),
        overall_metrics=dict(payload.get("overall_metrics") or {}),
        min_slice_size=int(payload.get("min_slice_size") or 1),
        numeric_bins=int(payload.get("numeric_bins") or 1),
        slices=tuple(payload.get("slices") or ()),
        table=pd.DataFrame(),
        worst_slices=tuple(payload.get("worst_slices") or ()),
    )


def _baseline_metrics(prepared, model_name: str) -> dict[str, float] | None:
    if prepared is None:
        return None
    metrics = prepared["metrics"]
    model_metrics = metrics.get(model_name) if isinstance(metrics, dict) else None
    if not isinstance(model_metrics, dict):
        return None
    return dict(model_metrics)


def _experiment_record(config, model_name, experiment_id, experiment_dir, reused, metadata):
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "dataset": config.dataset,
        "model": model_name,
        "models": list(config.models),
        "random_state": config.random_state,
        "test_size": config.test_size,
        "reused_artifacts": reused,
        "class_labels": metadata.get("class_labels"),
        "numeric_features": metadata.get("numeric_features"),
        "categorical_features": metadata.get("categorical_features"),
        "n_test_samples": metadata.get("n_test_samples"),
    }


def _selected_datasets(config: ExperimentConfig, datasets: Sequence[str] | None) -> tuple[str, ...]:
    if datasets is None:
        return (config.dataset,)
    if isinstance(datasets, str):
        raise TypeError("datasets must be a sequence of dataset names.")
    if len(datasets) == 0:
        raise ValueError("datasets must contain at least one dataset.")
    selected = tuple(validate_dataset(name) for name in datasets)
    if len(selected) != len(set(selected)):
        raise ValueError("datasets must not contain duplicate names.")
    return selected


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot read {path.name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def _identifiers(values) -> list:
    return [_identifier(value) for value in values]


def _identifier(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return str(value)
