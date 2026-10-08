"""Load classification datasets for benchmarking.

Datasets are fetched programmatically. ``load_dataset`` returns a stratified
split and an unfitted preprocessor. Fit that preprocessor from the training
workflow, on ``X_train`` only, so imputation and scaling never see ``X_test``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUPPORTED_DATASETS = ("breast_cancer", "adult")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPENML_DATA_HOME = _PROJECT_ROOT / "data" / "openml"
_MISSING_TOKENS = {"?", ""}
_ADULT_LABEL_TO_BINARY = {"<=50K": 0, ">50K": 1}


@dataclass(frozen=True, eq=False)
class DatasetSplit:
    """Stratified split and the preprocessor training should fit later.

    ``preprocessor`` is unfitted. Include it as a step in the model pipeline
    and fit that pipeline on ``X_train`` and ``y_train`` only.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    metadata: dict[str, Any]
    preprocessor: ColumnTransformer


def load_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Load a supported dataset and return a stratified train/test split.

    Parameters
    ----------
    name:
        ``"breast_cancer"`` or ``"adult"``.
    test_size:
        Fraction of rows assigned to the test split.
    random_state:
        Seed used by the stratified split.
    """
    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Supported names: {supported}.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if name == "breast_cancer":
        features, target, metadata = _load_breast_cancer()
        preprocessor_factory = _breast_cancer_preprocessor
    else:
        features, target, metadata = _load_adult()
        preprocessor_factory = _adult_preprocessor

    X_train, X_test, y_train, y_test = _stratified_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    numeric_features, categorical_features = _column_groups(X_train)
    preprocessor = preprocessor_factory(numeric_features, categorical_features)
    feature_names = [str(column) for column in X_train.columns]

    metadata = {
        **metadata,
        "n_samples": int(len(features)),
        "n_features": len(feature_names),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": test_size,
        "random_state": random_state,
        "stratified": True,
    }
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        metadata=metadata,
        preprocessor=preprocessor,
    )


def _load_breast_cancer() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    bunch = load_breast_cancer(as_frame=True)
    features = _with_string_columns(bunch.data)
    target = bunch.target.copy().astype("int64")
    class_labels = {
        index: str(label) for index, label in enumerate(bunch.target_names)
    }
    _validate_binary_target(target, dataset_name="breast_cancer")
    metadata = {
        "name": "breast_cancer",
        "source": "sklearn.datasets.load_breast_cancer",
        "target_name": str(target.name or "target"),
        "class_labels": class_labels,
    }
    return features, target, metadata


def _load_adult() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    _OPENML_DATA_HOME.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        "adult",
        version=2,
        as_frame=True,
        parser="auto",
        data_home=str(_OPENML_DATA_HOME),
    )
    features = _replace_missing_tokens(_with_string_columns(bunch.data))
    target = _adult_target_to_binary(bunch.target)
    metadata = {
        "name": "adult",
        "source": "sklearn.datasets.fetch_openml:adult:v2",
        "target_name": str(bunch.target.name or "class"),
        "class_labels": {0: "<=50K", 1: ">50K"},
    }
    return features, target, metadata


def _adult_target_to_binary(target: pd.Series) -> pd.Series:
    labels = target.astype(str).str.strip()
    unknown = sorted(set(labels.unique()) - set(_ADULT_LABEL_TO_BINARY))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unexpected Adult Income target labels: {joined}.")
    binary = labels.map(_ADULT_LABEL_TO_BINARY).astype("int64")
    binary.index = target.index
    binary.name = target.name
    return binary


def _with_string_columns(features: pd.DataFrame) -> pd.DataFrame:
    prepared = features.copy()
    prepared.columns = [str(column) for column in prepared.columns]
    return prepared


def _replace_missing_tokens(features: pd.DataFrame) -> pd.DataFrame:
    """Replace explicit missing markers without learning from the data."""
    prepared = features.copy()
    for column in prepared.columns:
        if pd.api.types.is_numeric_dtype(prepared[column]):
            continue
        text = prepared[column].astype("string").str.strip()
        missing = text.isna() | text.isin(list(_MISSING_TOKENS))
        values = pd.Series(np.nan, index=prepared.index, dtype=object)
        values.loc[~missing] = text.loc[~missing].astype(str)
        prepared[column] = values
    return prepared


def _column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        str(column)
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical = [
        str(column) for column in features.columns if column not in numeric
    ]
    return numeric, categorical


def _breast_cancer_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    if categorical_features:
        joined = ", ".join(categorical_features)
        raise ValueError(f"Breast Cancer features should be numeric, found: {joined}.")
    return ColumnTransformer(
        transformers=[("numeric", "passthrough", numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _adult_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    if not numeric_features or not categorical_features:
        raise ValueError(
            "Adult Income must contain both numeric and categorical features."
        )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _validate_binary_target(target, dataset_name="split")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def _validate_binary_target(target: pd.Series, dataset_name: str) -> None:
    classes = set(pd.Series(target).astype(int).unique())
    if classes != {0, 1}:
        raise ValueError(
            f"{dataset_name} target must be binary 0/1, found {sorted(classes)}."
        )
