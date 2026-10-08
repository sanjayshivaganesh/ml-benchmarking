"""Streamlit dashboard for one Phase 2 experiment.

This module is the UI. Dataset and model choices come from the registry.
The run uses the same configuration and ``run_phase2`` path as the command
line, and that path calls ``run_experiment``. Tables and plots come from the
visualization module.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.diagnostics.hard_examples import hard_examples_directory
from src.experiments.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    ExperimentConfig,
    apply_overrides,
    load_config,
)
from src.experiments.registry import list_datasets, list_models
from src.experiments.visualization import (
    metric_comparison_table,
    plot_confusion_matrices,
    plot_roc_curves,
)
from src.run_phase2 import Phase2Run, run_phase2

EXPLANATION = (
    "Run one classification experiment on a dataset and the models you select. "
    "Training, metrics, and error analysis are the same pipeline used by the "
    "command-line experiment. After the run, this page shows the class labels, "
    "metric table, ROC curves, and confusion matrices."
)


def dataset_choices() -> tuple[str, ...]:
    """Return the datasets the sidebar can offer."""
    return list_datasets()


def model_choices() -> tuple[str, ...]:
    """Return the models the sidebar can offer."""
    return list_models()


def experiment_config(
    dataset: str,
    models: list[str] | tuple[str, ...],
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    output_dir: Path | str | None = None,
) -> ExperimentConfig:
    """Validate a sidebar selection against the registry and configuration."""
    config = apply_overrides(
        load_config(),
        dataset=dataset,
        models=list(models),
        random_state=_seed(random_state),
        test_size=test_size,
    )
    if output_dir is not None:
        config = replace(config, output_dir=Path(output_dir).resolve())
    return config


def run_dashboard(
    dataset: str,
    models: list[str] | tuple[str, ...],
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    output_dir: Path | str | None = None,
) -> Phase2Run:
    """Run the shared experiment pipeline for the sidebar selection."""
    return run_phase2(
        experiment_config(
            dataset,
            models,
            random_state=random_state,
            test_size=test_size,
            output_dir=output_dir,
        )
    )


def class_label_lines(result) -> list[str]:
    """Return ``0 = name`` lines from the experiment result."""
    class_labels = getattr(result, "class_labels", None)
    if not isinstance(class_labels, dict) or not class_labels:
        raise ValueError("Class labels are missing from the experiment result.")
    lines = []
    for key in sorted(class_labels, key=lambda label: int(label)):
        lines.append(f"{int(key)} = {class_labels[key]}")
    return lines


def format_error(exc: BaseException) -> str:
    """Return a short message for the page, without a traceback."""
    if isinstance(exc, (ValueError, TypeError, FileNotFoundError, RuntimeError)):
        return str(exc)
    return f"Experiment failed: {exc}"


def main() -> None:
    """Draw the sidebar and the experiment page."""
    st.set_page_config(page_title="ML Experiment", layout="wide")
    st.title("ML Experiment")
    dataset, models, random_state, test_size = _sidebar()
    if st.button("Run Experiment"):
        try:
            st.session_state["outcome"] = run_dashboard(
                dataset,
                models,
                random_state=random_state,
                test_size=test_size,
            )
            st.session_state["error"] = None
        except Exception as exc:
            st.session_state["outcome"] = None
            st.session_state["error"] = format_error(exc)

    error = st.session_state.get("error")
    outcome = st.session_state.get("outcome")
    if error:
        st.error(error)
    if outcome is None:
        st.write(EXPLANATION)
        return
    _show(outcome)


def _sidebar() -> tuple[str, list[str], int, float]:
    st.sidebar.header("Experiment")
    dataset = st.sidebar.selectbox("Dataset", dataset_choices())
    models = st.sidebar.multiselect("Models", model_choices())
    random_state = st.sidebar.number_input(
        "Random state",
        min_value=0,
        value=DEFAULT_RANDOM_STATE,
        step=1,
    )
    test_size = st.sidebar.number_input(
        "Test size",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_TEST_SIZE,
        step=0.05,
        format="%.2f",
    )
    return str(dataset), list(models), _seed(random_state), float(test_size)


def _show(outcome: Phase2Run) -> None:
    result = outcome.result
    st.header("Experiment configuration")
    st.write(f"Dataset: {result.dataset}")
    st.write(f"Models: {', '.join(result.models)}")
    st.write(f"random_state: {result.random_state}")
    st.write(f"test_size: {result.test_size}")
    st.write(f"experiment_id: {outcome.experiment_id}")

    st.header("Class labels")
    for line in class_label_lines(result):
        st.write(line)

    st.header("Metric comparison")
    st.dataframe(metric_comparison_table(result))

    st.header("ROC curve")
    roc_axis = plot_roc_curves(result)
    st.pyplot(roc_axis.figure)
    plt.close(roc_axis.figure)

    st.header("Confusion matrices")
    figure = plot_confusion_matrices(result)
    st.pyplot(figure)
    plt.close(figure)

    st.header("Hard examples")
    _show_hard_examples(outcome)

    st.header("Outputs")
    st.write(str(outcome.output_dir))
    st.write(str(hard_examples_directory(outcome.output_dir)))


def _show_hard_examples(outcome: Phase2Run) -> None:
    """Show the hard-example files already written for this experiment."""
    directory = hard_examples_directory(outcome.output_dir)
    for model_name in outcome.result.models:
        path = directory / f"{outcome.result.dataset}_{model_name}_hard_examples.json"
        st.subheader(model_name)
        if not path.is_file():
            st.write(f"No hard-example file at {path}.")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        st.write(
            "Test rows: {n_samples}. "
            "Correct and confident: {correct_confident}. "
            "Correct and uncertain: {correct_uncertain}. "
            "Wrong and confident: {wrong_confident}. "
            "Wrong and uncertain: {wrong_uncertain}.".format(
                n_samples=summary.get("n_samples", 0),
                correct_confident=summary.get("correct_confident", 0),
                correct_uncertain=summary.get("correct_uncertain", 0),
                wrong_confident=summary.get("wrong_confident", 0),
                wrong_uncertain=summary.get("wrong_uncertain", 0),
            )
        )
        st.write("High-confidence errors")
        st.dataframe(hard_example_table(payload.get("high_confidence_errors", [])))
        st.write("Most uncertain")
        st.dataframe(hard_example_table(payload.get("most_uncertain", [])))


def hard_example_table(rows) -> pd.DataFrame:
    """Flatten hard-example records, including original feature values."""
    records = []
    for row in rows:
        record = {
            key: row.get(key)
            for key in (
                "sample_id",
                "y_true",
                "y_pred",
                "probability",
                "confidence",
                "uncertainty",
                "correct",
            )
        }
        features = row.get("features") or {}
        record.update(features)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _seed(value) -> int:
    if isinstance(value, bool):
        raise ValueError("random_state must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("random_state must be an integer.")


if __name__ == "__main__":
    main()
