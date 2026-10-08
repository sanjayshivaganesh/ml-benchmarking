"""Train baseline models on each benchmark dataset.

Each run fits one sklearn Pipeline, preprocessing then classifier, on the
training split only and writes the fitted pipeline to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.data.data_loader import SUPPORTED_DATASETS, DatasetSplit, load_dataset
from src.models.models import MODEL_NAMES, get_models

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"


@dataclass(frozen=True, eq=False)
class TrainedModel:
    """Provenance for one saved dataset/model pipeline."""

    dataset: str
    model: str
    path: Path
    metadata: dict[str, Any]


def train_all(
    datasets: list[str] | tuple[str, ...] | None = None,
    random_state: int = 42,
    model_dir: Path | str | None = None,
) -> list[TrainedModel]:
    """Train every selected dataset against every baseline model.

    Parameters
    ----------
    datasets:
        Dataset names understood by ``load_dataset``. Defaults to all of them.
    random_state:
        Seed forwarded to the dataset split and the classifiers.
    model_dir:
        Directory for fitted pipelines. Defaults to the project ``models`` folder.
    """
    selected = _resolve_datasets(datasets)
    output_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations = [
        (dataset_name, model_name)
        for dataset_name in selected
        for model_name in MODEL_NAMES
    ]

    trained: list[TrainedModel] = []
    for dataset_name, model_name in tqdm(combinations, desc="Training models"):
        trained.append(
            train_one(
                dataset=dataset_name,
                model=model_name,
                random_state=random_state,
                test_size=0.2,
                model_dir=output_dir,
            )
        )
    return trained


def train_one(
    dataset: str,
    model: str,
    random_state: int = 42,
    test_size: float = 0.2,
    model_dir: Path | str | None = None,
) -> TrainedModel:
    """Train one baseline model on one dataset.

    This is the single-combination form of ``train_all``. The split stays
    stratified. ``test_size`` defaults to 0.2, and ``random_state`` is
    forwarded to that split and the classifier. The fitted object is the same
    preprocessing-then-classifier pipeline, with the same metadata and
    ``{dataset}__{model}.joblib`` filename.

    Parameters
    ----------
    dataset:
        Dataset name understood by ``load_dataset``.
    model:
        Model name returned by ``get_models``.
    random_state:
        Seed forwarded to the dataset split and the classifier.
    test_size:
        Fraction of rows assigned to the test split.
    model_dir:
        Directory for the fitted pipeline. Defaults to the project ``models`` folder.
    """
    dataset_name = _resolve_datasets((dataset,))[0]
    model_name = _resolve_model(model)
    output_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = clone(get_models(random_state=random_state)[model_name])
    return _train_one(
        dataset_name=dataset_name,
        model_name=model_name,
        estimator=estimator,
        random_state=random_state,
        test_size=test_size,
        model_dir=output_dir,
    )


def _train_one(
    dataset_name: str,
    model_name: str,
    estimator,
    random_state: int,
    model_dir: Path,
    test_size: float = 0.2,
) -> TrainedModel:
    split = load_dataset(
        dataset_name,
        test_size=test_size,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", clone(split.preprocessor)),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(split.X_train, split.y_train)

    path = model_dir / f"{dataset_name}__{model_name}.joblib"
    metadata = _experiment_metadata(
        dataset_name=dataset_name,
        model_name=model_name,
        random_state=random_state,
        split=split,
        estimator=pipeline.named_steps["classifier"],
        path=path,
    )
    pipeline.metadata = metadata
    joblib.dump(pipeline, path)
    return TrainedModel(
        dataset=dataset_name,
        model=model_name,
        path=path,
        metadata=metadata,
    )


def _experiment_metadata(
    dataset_name: str,
    model_name: str,
    random_state: int,
    split: DatasetSplit,
    estimator,
    path: Path,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "model": model_name,
        "random_state": random_state,
        "test_size": split.metadata["test_size"],
        "source": split.metadata["source"],
        "class_labels": dict(split.metadata["class_labels"]),
        "feature_names": list(split.feature_names),
        "numeric_features": list(split.metadata["numeric_features"]),
        "categorical_features": list(split.metadata["categorical_features"]),
        "n_train_samples": int(len(split.y_train)),
        "stratified": bool(split.metadata["stratified"]),
        "hyperparameters": estimator.get_params(),
        "sklearn_version": sklearn.__version__,
        "filename": path.name,
    }


def _resolve_model(model: str) -> str:
    if model not in MODEL_NAMES:
        supported = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {model!r}. Supported names: {supported}.")
    return model


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
