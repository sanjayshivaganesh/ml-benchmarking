"""Build the Phase 1 Markdown summary from pipeline artifacts."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.analysis.error_analysis import DEFAULT_ERROR_REPORT_PATH, analyze_all
from src.data.data_loader import SUPPORTED_DATASETS
from src.evaluation.evaluate import DEFAULT_METRICS_PATH, METRIC_NAMES, evaluate_all
from src.models.models import MODEL_NAMES
from src.training.train import DEFAULT_MODEL_DIR, TrainedModel, train_all

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_PATH = _PROJECT_ROOT / "outputs" / "reports" / "summary.md"

NOTABLE_MIN_COUNT = 20
NOTABLE_RATE_GAP = 0.05
_DISPLAY_DECIMALS = 4
_HYPERPARAMETER_KEYS = (
    "C",
    "max_iter",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "random_state",
)
_REQUIRED_METADATA = (
    "source",
    "class_labels",
    "numeric_features",
    "categorical_features",
    "n_train_samples",
    "random_state",
    "test_size",
    "hyperparameters",
)


def run_phase1(
    random_state: int = 42,
    model_dir: Path | str | None = None,
    metrics_path: Path | str | None = None,
    error_report_path: Path | str | None = None,
    summary_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train, evaluate, analyze errors, and write ``summary.md``.

    The summary is rendered from the metrics, error report, and saved
    pipeline metadata produced by this call. ``random_state`` is forwarded
    to the split and the classifiers.
    """
    artifact_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    metrics_output = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH
    error_output = (
        Path(error_report_path) if error_report_path is not None else DEFAULT_ERROR_REPORT_PATH
    )
    summary_output = Path(summary_path) if summary_path is not None else DEFAULT_SUMMARY_PATH

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained = train_all(random_state=random_state, model_dir=artifact_dir)
        metrics = evaluate_all(
            model_dir=artifact_dir,
            metrics_path=metrics_output,
            random_state=random_state,
        )
        error_report = analyze_all(
            model_dir=artifact_dir,
            report_path=error_output,
            random_state=random_state,
        )
        runtime_warnings = _unique_warnings(caught)

    setup = build_experiment_setup(trained, error_report, runtime_warnings)
    summary = render_summary(metrics, error_report, setup)
    _write_summary(summary, summary_output)
    return {
        "metrics": metrics,
        "error_report": error_report,
        "setup": setup,
        "summary": summary,
        "summary_path": summary_output,
        "metrics_path": metrics_output,
        "error_report_path": error_output,
        "model_dir": artifact_dir,
    }


def render_summary(
    metrics: dict[str, dict[str, dict[str, float]]],
    error_report: dict[str, dict[str, dict[str, Any]]],
    setup: dict[str, Any],
) -> str:
    """Render the Phase 1 report from in-memory pipeline results."""
    _validate_results(metrics, error_report)
    datasets = _ordered(metrics, SUPPORTED_DATASETS)
    lines = [
        "# ML Benchmarking — Phase 1",
        "",
        "## Experimental setup",
        "",
        *_dataset_lines(setup, datasets),
        "",
        *_model_lines(setup, datasets, metrics),
        "",
        *_split_lines(setup),
        "",
        *_preprocessing_lines(setup, datasets),
        "",
        "## Overall results",
        "",
        (
            f"Table values are rounded to {_DISPLAY_DECIMALS} decimal places. "
            "Unrounded values are in `outputs/metrics/metrics.json`."
        ),
        "",
        *_metrics_table(metrics, datasets),
        "",
        "## Best-performing models",
        "",
        (
            "Within each dataset, the best model is the one with the highest F1. "
            "ROC-AUC is reported alongside F1 and is not used to break ties."
        ),
        "",
        *_best_model_lines(metrics, datasets),
        "",
        "## Failure analysis",
        "",
        (
            "A slice level is elevated when it has at least "
            f"{NOTABLE_MIN_COUNT} test samples and its error rate is at least "
            f"{_format_metric(NOTABLE_RATE_GAP)} above that model's "
            "misclassification rate."
        ),
        "",
        *_failure_lines(error_report, datasets),
        "### Limitations",
        "",
        *_limitation_lines(),
        "",
        "## Warnings",
        "",
        *_warning_lines(setup.get("warnings", [])),
        "",
    ]
    return "\n".join(lines)


def build_experiment_setup(
    trained: list[TrainedModel],
    error_report: dict[str, dict[str, dict[str, Any]]],
    runtime_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Collect setup facts from saved pipelines and the error report."""
    if not trained:
        raise ValueError("Cannot describe an experiment with no trained models.")
    notes: list[str] = list(runtime_warnings or [])
    by_dataset: dict[str, list[TrainedModel]] = {}
    for item in trained:
        by_dataset.setdefault(item.dataset, []).append(item)

    dataset_setup: dict[str, dict[str, Any]] = {}
    model_parameters: dict[str, dict[str, Any]] = {}
    for dataset_name, items in by_dataset.items():
        dataset_setup[dataset_name] = _dataset_setup(items, error_report, notes)
        for item in items:
            parameters = _selected_hyperparameters(item.metadata["hyperparameters"])
            previous = model_parameters.get(item.model)
            if previous is None:
                model_parameters[item.model] = parameters
            elif previous != parameters:
                notes.append(
                    f"{item.model} hyperparameters differ across datasets; "
                    f"the report lists the values from the first saved artifact."
                )

    first = trained[0].metadata
    random_states = {item.metadata["random_state"] for item in trained}
    test_sizes = {item.metadata["test_size"] for item in trained}
    stratified_flags = {
        item.metadata.get("stratified")
        for item in trained
        if "stratified" in item.metadata
    }
    if len(random_states) != 1:
        notes.append(f"Saved artifacts do not share one random_state: {sorted(random_states)}.")
    if len(test_sizes) != 1:
        notes.append(f"Saved artifacts do not share one test_size: {sorted(test_sizes)}.")
    if len(stratified_flags) > 1:
        notes.append(f"Saved artifacts disagree on stratified splitting: {sorted(stratified_flags)}.")

    return {
        "random_state": first["random_state"],
        "test_size": first["test_size"],
        "stratified": first.get("stratified"),
        "sklearn_version": first.get("sklearn_version"),
        "datasets": dataset_setup,
        "models": model_parameters,
        "warnings": _dedupe(notes),
    }


def describe_preprocessor(preprocessor) -> str:
    """Describe a fitted or unfitted column transformer from its steps."""
    transformers = getattr(preprocessor, "transformers_", None)
    if transformers is None:
        transformers = getattr(preprocessor, "transformers", ())
    parts = []
    for name, transformer, columns in transformers:
        if transformer == "drop":
            continue
        count = _column_count(columns)
        parts.append(f"{name} ({count} columns): {_transformer_text(transformer)}")
    if not parts:
        return "No preprocessing steps were recorded."
    return "; ".join(parts)


def _dataset_setup(
    items: list[TrainedModel],
    error_report: dict[str, dict[str, dict[str, Any]]],
    notes: list[str],
) -> dict[str, Any]:
    first = items[0]
    _require_metadata(first)
    metadata = first.metadata
    descriptions = {}
    for item in items:
        _require_metadata(item)
        pipeline = joblib.load(item.path)
        descriptions[item.model] = describe_preprocessor(
            pipeline.named_steps["preprocessing"]
        )
    unique_descriptions = set(descriptions.values())
    if len(unique_descriptions) != 1:
        notes.append(
            f"{first.dataset} preprocessing descriptions differ across models: "
            + "; ".join(f"{model}: {text}" for model, text in descriptions.items())
        )
    n_test_values = {
        int(error_report[first.dataset][item.model]["n_test"]) for item in items
    }
    if len(n_test_values) != 1:
        notes.append(
            f"{first.dataset} error report has different test sizes: {sorted(n_test_values)}."
        )
    n_train_values = {int(item.metadata["n_train_samples"]) for item in items}
    if len(n_train_values) != 1:
        notes.append(
            f"{first.dataset} artifacts have different training sizes: {sorted(n_train_values)}."
        )
    return {
        "source": metadata["source"],
        "class_labels": dict(metadata["class_labels"]),
        "n_train": int(metadata["n_train_samples"]),
        "n_test": int(next(iter(n_test_values))),
        "n_numeric_features": len(metadata["numeric_features"]),
        "n_categorical_features": len(metadata["categorical_features"]),
        "preprocessing": descriptions[first.model],
    }


def _dataset_lines(setup: dict[str, Any], datasets: list[str]) -> list[str]:
    lines = ["### Datasets", ""]
    described = setup["datasets"]
    for name in datasets:
        info = described[name]
        lines.append(
            f"- `{name}` from `{info['source']}`. "
            f"Labels: {_class_label_text(info['class_labels'])}. "
            f"{info['n_train']} train rows, {info['n_test']} test rows. "
            f"{_plural(info['n_numeric_features'], 'numeric feature')}, "
            f"{_plural(info['n_categorical_features'], 'categorical feature')}."
        )
    return lines


def _model_lines(
    setup: dict[str, Any],
    datasets: list[str],
    metrics: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    lines = [
        "### Models",
        "",
        "The same baseline hyperparameters are used for every dataset. They are not tuned.",
        "",
    ]
    model_names = _ordered(_model_names(metrics, datasets), MODEL_NAMES)
    recorded = setup["models"]
    for name in model_names:
        parameters = recorded.get(name, {})
        if parameters:
            rendered = ", ".join(f"{key}={parameters[key]}" for key in parameters)
            lines.append(f"- `{name}`: {rendered}.")
        else:
            lines.append(f"- `{name}`.")
    return lines


def _split_lines(setup: dict[str, Any]) -> list[str]:
    lines = [
        "### Train/test split",
        "",
        f"- test_size: {setup['test_size']}",
        f"- random_state: {setup['random_state']}",
    ]
    stratified = setup.get("stratified")
    if stratified is True:
        lines.append("- splitting: stratified by the binary target")
    elif stratified is False:
        lines.append("- splitting: not stratified")
    lines.append(
        "- precision, recall, and F1 use predicted labels at the estimator's default 0.5 threshold"
    )
    lines.append("- ROC-AUC and Brier score use the predicted probability of class 1")
    version = setup.get("sklearn_version")
    if version:
        lines.append(f"- scikit-learn: {version}")
    return lines


def _preprocessing_lines(setup: dict[str, Any], datasets: list[str]) -> list[str]:
    lines = [
        "### Preprocessing",
        "",
        "Each saved artifact is one pipeline: preprocessing, then the classifier.",
        "",
    ]
    for name in datasets:
        lines.append(f"- `{name}`: {setup['datasets'][name]['preprocessing']}.")
    return lines


def _metrics_table(
    metrics: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
) -> list[str]:
    header = (
        "| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |"
    )
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for dataset_name in datasets:
        for model_name in _ordered(metrics[dataset_name], MODEL_NAMES):
            scores = metrics[dataset_name][model_name]
            values = " | ".join(_format_metric(scores[name]) for name in METRIC_NAMES)
            rows.append(f"| {dataset_name} | {model_name} | {values} |")
    return rows


def _best_model_lines(
    metrics: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
) -> list[str]:
    lines = []
    for dataset_name in datasets:
        scores = metrics[dataset_name]
        f1_winners = _leaders(scores, "f1")
        auc_winners = _leaders(scores, "roc_auc")
        f1_text = _join_names(f1_winners)
        auc_text = _join_names(auc_winners)
        winner_details = "; ".join(
            f"`{name}` F1 {_format_metric(scores[name]['f1'])}, "
            f"ROC-AUC {_format_metric(scores[name]['roc_auc'])}"
            for name in f1_winners
        )
        lines.append(f"- `{dataset_name}` highest F1: {winner_details}.")
        if f1_winners == auc_winners:
            lines.append(f"  The same model selection has the highest ROC-AUC: {auc_text}.")
        else:
            auc_details = "; ".join(
                f"`{name}` ROC-AUC {_format_metric(scores[name]['roc_auc'])}, "
                f"F1 {_format_metric(scores[name]['f1'])}"
                for name in auc_winners
            )
            lines.append(
                f"  Highest ROC-AUC is {auc_text}, which is not the highest-F1 selection: {auc_details}."
            )
    return lines


def _failure_lines(
    error_report: dict[str, dict[str, dict[str, Any]]],
    datasets: list[str],
) -> list[str]:
    lines = [
        "| Dataset | Model | False positives | False negatives | Misclassified | Error rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_name in datasets:
        for model_name in _ordered(error_report[dataset_name], MODEL_NAMES):
            report = error_report[dataset_name][model_name]
            lines.append(
                "| {dataset} | {model} | {fp} | {fn} | {count} | {rate} |".format(
                    dataset=dataset_name,
                    model=model_name,
                    fp=int(report["false_positives"]),
                    fn=int(report["false_negatives"]),
                    count=int(report["misclassified_count"]),
                    rate=_format_metric(report["misclassification_rate"]),
                )
            )
    lines.append("")
    for dataset_name in datasets:
        lines.extend(_dataset_failure_lines(dataset_name, error_report[dataset_name]))
        lines.append("")
    return lines


def _dataset_failure_lines(dataset_name: str, reports: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"### {dataset_name}", ""]
    most_errors = _count_leaders(reports, "misclassified_count")
    count = int(reports[most_errors[0]]["misclassified_count"])
    n_test = int(reports[most_errors[0]]["n_test"])
    lines.append(
        f"{_join_names(most_errors)} misclassified the most {dataset_name} test rows: "
        f"{count} of {n_test}."
    )
    for model_name in _ordered(reports, MODEL_NAMES):
        report = reports[model_name]
        false_positives = int(report["false_positives"])
        false_negatives = int(report["false_negatives"])
        lines.append(
            f"- `{model_name}`: {_plural(false_positives, 'false positive')} and "
            f"{_plural(false_negatives, 'false negative')}. "
            f"{_error_balance(false_positives, false_negatives)}"
        )
        if int(report["misclassified_count"]) != false_positives + false_negatives:
            lines.append(
                f"  `misclassified_count` ({int(report['misclassified_count'])}) "
                "does not equal false positives plus false negatives."
            )
        elevated = _elevated_slices(report)
        if elevated:
            lines.append("  Elevated slices:")
            for item in elevated:
                lines.append(
                    "  - `{feature}` / `{level}`: error rate {rate} "
                    "({errors} / {count}), model misclassification rate {overall}.".format(
                        feature=item["feature"],
                        level=item["level"],
                        rate=_format_metric(item["error_rate"]),
                        errors=item["errors"],
                        count=item["count"],
                        overall=_format_metric(report["misclassification_rate"]),
                    )
                )
        else:
            lines.append("  No slice level met the elevation rule.")
    return lines


def _elevated_slices(report: dict[str, Any]) -> list[dict[str, Any]]:
    overall = float(report["misclassification_rate"])
    found = []
    for feature_name, spec in report.get("feature_slices", {}).items():
        for level_name, row in _slice_levels(spec):
            count = int(row["count"])
            error_rate = float(row["error_rate"])
            if count < NOTABLE_MIN_COUNT:
                continue
            if error_rate < overall + NOTABLE_RATE_GAP:
                continue
            found.append(
                {
                    "feature": feature_name,
                    "level": level_name,
                    "count": count,
                    "errors": int(row["errors"]),
                    "error_rate": error_rate,
                }
            )
    found.sort(key=lambda item: (-item["error_rate"], item["feature"], str(item["level"])))
    return found


def _slice_levels(spec: dict[str, Any]):
    if spec.get("type") == "numeric":
        for row in spec.get("bins", []):
            yield row["bin"], row
    elif spec.get("type") == "categorical":
        for row in spec.get("categories", []):
            yield row["category"], row


def _error_balance(false_positives: int, false_negatives: int) -> str:
    if false_negatives > false_positives:
        return (
            f"False negatives exceed false positives by {false_negatives - false_positives}."
        )
    if false_positives > false_negatives:
        return (
            f"False positives exceed false negatives by {false_positives - false_negatives}."
        )
    return "False positives and false negatives are equal."


def _limitation_lines() -> list[str]:
    return [
        "- Results come from one holdout split. They are not cross-validated.",
        "- Hyperparameters are fixed baselines. Thresholds are not tuned; class labels use the estimator default of 0.5.",
        "- Each model slices at most two original features. The features are chosen by their association with errors on this same test set, so the slices describe the test errors rather than a separate confirmation set.",
        "- Numeric slices use up to four quantile bins. Categorical levels below the support cutoff stored in `error_report.json` are omitted.",
        f"- Metrics in the table are rounded to {_DISPLAY_DECIMALS} decimal places.",
    ]


def _warning_lines(messages: list[str]) -> list[str]:
    if not messages:
        return ["No warnings were raised during this run."]
    return [f"- {message}" for message in messages]


def _validate_results(
    metrics: dict[str, dict[str, dict[str, float]]],
    error_report: dict[str, dict[str, dict[str, Any]]],
) -> None:
    if set(metrics) != set(error_report):
        raise ValueError("Metrics and error report dataset keys do not match.")
    for dataset_name, models in metrics.items():
        if set(models) != set(error_report[dataset_name]):
            raise ValueError(f"{dataset_name} model keys differ between metrics and errors.")
        for model_name, scores in models.items():
            missing = [name for name in METRIC_NAMES if name not in scores]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"{dataset_name}/{model_name} is missing metrics: {joined}.")
            for name in METRIC_NAMES:
                value = float(scores[name])
                if value != value or value in {float("inf"), float("-inf")}:
                    raise ValueError(f"{dataset_name}/{model_name} {name} is not finite.")
            report = error_report[dataset_name][model_name]
            for field in (
                "false_positives",
                "false_negatives",
                "misclassified_count",
                "misclassification_rate",
                "n_test",
                "feature_slices",
            ):
                if field not in report:
                    raise ValueError(f"{dataset_name}/{model_name} error report lacks {field}.")


def _leaders(scores: dict[str, dict[str, float]], metric_name: str) -> list[str]:
    best = max(float(metrics[metric_name]) for metrics in scores.values())
    return [
        name
        for name in _ordered(scores, MODEL_NAMES)
        if float(scores[name][metric_name]) == best
    ]


def _count_leaders(reports: dict[str, dict[str, Any]], field: str) -> list[str]:
    best = max(int(report[field]) for report in reports.values())
    return [
        name
        for name in _ordered(reports, MODEL_NAMES)
        if int(reports[name][field]) == best
    ]


def _selected_hyperparameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: parameters[key]
        for key in _HYPERPARAMETER_KEYS
        if key in parameters and parameters[key] is not None
    }


def _require_metadata(item: TrainedModel) -> None:
    missing = [key for key in _REQUIRED_METADATA if key not in item.metadata]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{item.dataset}/{item.model} metadata lacks {joined}.")


def _transformer_text(transformer) -> str:
    if transformer == "passthrough" or _is_passthrough(transformer):
        return "passthrough"
    if isinstance(transformer, Pipeline):
        return ", ".join(
            _transformer_text(step) for _name, step in transformer.steps
        )
    if isinstance(transformer, SimpleImputer):
        strategy = str(transformer.strategy).replace("_", "-")
        return f"{strategy} imputation"
    if isinstance(transformer, StandardScaler):
        return "standard scaling"
    if isinstance(transformer, OneHotEncoder):
        return "one-hot encoding"
    return type(transformer).__name__


def _is_passthrough(transformer) -> bool:
    return isinstance(transformer, FunctionTransformer) and transformer.func is None


def _column_count(columns) -> int:
    if columns is None or isinstance(columns, str):
        return 0
    try:
        return len(columns)
    except TypeError:
        return 0


def _class_label_text(class_labels: dict[Any, Any]) -> str:
    keys = sorted(class_labels, key=lambda key: int(key))
    return ", ".join(f"{int(key)} = {class_labels[key]}" for key in keys)


def _model_names(
    metrics: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
) -> list[str]:
    names = []
    for dataset_name in datasets:
        for model_name in metrics[dataset_name]:
            if model_name not in names:
                names.append(model_name)
    return names


def _ordered(names, preferred: tuple[str, ...] | list[str]) -> list[str]:
    present = list(names)
    first = [name for name in preferred if name in present]
    rest = sorted(name for name in present if name not in first)
    return first + rest


def _join_names(names: list[str]) -> str:
    rendered = [f"`{name}`" for name in names]
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def _plural(count: int, singular: str) -> str:
    word = singular if int(count) == 1 else f"{singular}s"
    return f"{int(count)} {word}"


def _format_metric(value: float) -> str:
    return f"{float(value):.{_DISPLAY_DECIMALS}f}"


def _unique_warnings(caught) -> list[str]:
    messages = [f"{warning.category.__name__}: {warning.message}" for warning in caught]
    return _dedupe(messages)


def _dedupe(messages: list[str]) -> list[str]:
    unique = []
    for message in messages:
        if message not in unique:
            unique.append(message)
    return unique


def _write_summary(summary: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
