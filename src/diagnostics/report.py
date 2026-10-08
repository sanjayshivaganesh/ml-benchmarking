"""Render a Phase 3 failure report from a diagnostic-engine result.

The report copies stored metrics, slices, examples, robustness values, and
importance ranks. It does not score predictions again and does not train a
model. Statements in the summary are fixed templates. They describe stored
associations and say when evidence is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.diagnostics.diagnostic_engine import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_NOT_SUPPORTED,
    STATUS_SUPPORTED,
    DiagnosticEngineResult,
)
from src.diagnostics.model_comparison import (
    DEFAULT_MODEL_COMPARISON_PATH,
    save_model_comparison,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAILURE_ANALYSIS_PATH = _PROJECT_ROOT / "outputs" / "diagnostics" / "failure_analysis.md"
DEFAULT_ROBUSTNESS_REPORT_PATH = (
    _PROJECT_ROOT / "outputs" / "diagnostics" / "robustness_report.json"
)
_DISPLAY_DECIMALS = 4
_ASSOCIATION_NOTE = (
    "These differences are performance associations within the stored test set. "
    "They do not show that a feature caused the model to fail."
)
_PERFORMANCE_METRICS = (
    ("accuracy", "accuracy"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
)
_SUMMARY_COUNTS = (
    ("n_samples", "test rows"),
    ("n_errors", "incorrect predictions"),
    ("correct_confident", "correct and confident"),
    ("correct_uncertain", "correct and uncertain"),
    ("wrong_confident", "incorrect and confident"),
    ("wrong_uncertain", "incorrect and uncertain"),
)


@dataclass(frozen=True)
class DiagnosticReportPaths:
    """Locations of the written markdown report, robustness JSON, and model comparison."""

    markdown_path: Path
    robustness_path: Path
    comparison_path: Path


def render_failure_analysis(result: DiagnosticEngineResult | dict[str, Any]) -> str:
    """Return the human-readable failure report for one diagnostic result."""
    payload = _payload(result)
    sections = [
        "# Model Failure Analysis",
        "",
        "This report is rendered from stored diagnostic results. "
        "Displayed numbers use 4 decimal places. "
        f"The robustness JSON at `{DEFAULT_ROBUSTNESS_REPORT_PATH.name}` "
        "keeps the stored values.",
        "",
        _experiment_section(payload),
        _performance_section(payload),
        _slices_section(payload),
        _high_confidence_section(payload),
        _uncertain_section(payload),
        _robustness_section(payload),
        _importance_section(payload),
        _summary_section(payload),
    ]
    return "\n".join(sections).rstrip() + "\n"


def build_robustness_report(result: DiagnosticEngineResult | dict[str, Any]) -> dict[str, Any]:
    """Copy stored robustness results into one machine-readable document."""
    payload = _payload(result)
    runs = []
    comparison = []
    for run in _runs(payload):
        experiment = _experiment(run)
        component = _component(run, "robustness")
        stored = _stored(component)
        levels = [] if stored is None else list(stored.get("results") or [])
        runs.append(
            {
                "dataset": experiment.get("dataset"),
                "model": experiment.get("model"),
                "status": component.get("status"),
                "detail": component.get("detail"),
                "baseline_metrics": None if stored is None else stored.get("baseline_metrics"),
                "noise_configuration": None if stored is None else stored.get("noise_configuration"),
                "results": levels,
            }
        )
        for level in levels:
            if not isinstance(level, dict):
                continue
            mean = level.get("mean") if isinstance(level.get("mean"), dict) else {}
            std = level.get("std") if isinstance(level.get("std"), dict) else {}
            degradation = (
                level.get("degradation") if isinstance(level.get("degradation"), dict) else {}
            )
            comparison.append(
                {
                    "dataset": experiment.get("dataset"),
                    "model": experiment.get("model"),
                    "noise_level": level.get("noise_level"),
                    "f1_mean": mean.get("f1"),
                    "f1_std": std.get("f1"),
                    "f1_drop": degradation.get("f1"),
                }
            )
    return {
        "datasets": list(payload.get("datasets") or []),
        "models": list(payload.get("models") or []),
        "random_state": payload.get("random_state"),
        "test_size": payload.get("test_size"),
        "runs": runs,
        "comparison": comparison,
    }


def write_diagnostic_reports(
    result: DiagnosticEngineResult | dict[str, Any],
    *,
    directory: Path | str | None = None,
    markdown_path: Path | str | None = None,
    robustness_path: Path | str | None = None,
    comparison_path: Path | str | None = None,
) -> DiagnosticReportPaths:
    """Write the failure report, robustness JSON, and model comparison."""
    markdown = render_failure_analysis(result)
    robustness = build_robustness_report(result)
    markdown_destination = _destination(
        markdown_path,
        directory,
        DEFAULT_FAILURE_ANALYSIS_PATH,
    )
    robustness_destination = _destination(
        robustness_path,
        directory,
        DEFAULT_ROBUSTNESS_REPORT_PATH,
    )
    comparison_destination = _destination(
        comparison_path,
        directory,
        DEFAULT_MODEL_COMPARISON_PATH,
    )
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    robustness_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text(markdown, encoding="utf-8")
    with robustness_destination.open("w", encoding="utf-8") as handle:
        json.dump(robustness, handle, indent=2, allow_nan=False)
        handle.write("\n")
    save_model_comparison(result, comparison_destination)
    return DiagnosticReportPaths(
        markdown_destination,
        robustness_destination,
        comparison_destination,
    )


def _experiment_section(payload: dict[str, Any]) -> str:
    lines = ["## Experiment", ""]
    lines.append(f"- datasets: {_joined(payload.get('datasets'))}")
    lines.append(f"- models: {_joined(payload.get('models'))}")
    if "random_state" in payload:
        lines.append(f"- random_state: {payload.get('random_state')}")
    if "test_size" in payload:
        lines.append(f"- test_size: {payload.get('test_size')}")
    runs = _runs(payload)
    if not runs:
        lines.extend(["", "No model diagnostics were included."])
        return "\n".join(lines)
    for run in runs:
        experiment = _experiment(run)
        lines.extend(["", f"### {_model_heading(experiment)}", ""])
        for key, label in (
            ("experiment_id", "experiment id"),
            ("random_state", "random_state"),
            ("test_size", "test_size"),
            ("n_test_samples", "test rows"),
            ("reused_artifacts", "reused artifacts"),
        ):
            if experiment.get(key) is not None:
                lines.append(f"- {label}: {experiment.get(key)}")
        class_labels = experiment.get("class_labels")
        if isinstance(class_labels, dict) and class_labels:
            lines.append(f"- class labels: {_class_label_text(class_labels)}")
        for key, label in (
            ("numeric_features", "numeric features"),
            ("categorical_features", "categorical features"),
        ):
            values = experiment.get(key)
            if isinstance(values, list) and values:
                lines.append(f"- {label}: {', '.join(str(value) for value in values)}")
    return "\n".join(lines)


def _performance_section(payload: dict[str, Any]) -> str:
    lines = [
        "## Overall Performance",
        "",
        "Accuracy, precision, recall, F1, and ROC-AUC are copied from the stored baseline.",
        "",
    ]
    rows = []
    for run in _runs(payload):
        experiment = _experiment(run)
        metrics = run.get("baseline_metrics")
        stored = metrics if isinstance(metrics, dict) else {}
        rows.append(
            [
                experiment.get("dataset"),
                experiment.get("model"),
                *[_metric(stored, key) for key, _label in _PERFORMANCE_METRICS],
            ]
        )
    if not rows:
        lines.append("There is insufficient evidence to report overall performance.")
        return "\n".join(lines)
    lines.append(
        _table(
            ["dataset", "model", *[label for _key, label in _PERFORMANCE_METRICS]],
            rows,
        )
    )
    return "\n".join(lines)


def _slices_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 1. Failure Slices",
        "",
        "The table lists the worst-performing valid slices stored by the diagnostic engine. "
        "F1 is the ranking metric. Accuracy is included because its delta is already stored.",
        "",
        _ASSOCIATION_NOTE,
        "",
    ]
    lines.extend(_model_blocks(payload, _slice_block))
    return "\n".join(lines).rstrip()


def _slice_block(run: dict[str, Any]) -> list[str]:
    component = _component(run, "slicing")
    lines = [_status_line(component)]
    stored = _stored(component)
    if stored is None:
        lines.append("There is insufficient evidence to list failure slices.")
        return lines
    worst = [item for item in stored.get("worst_slices") or [] if isinstance(item, dict)]
    note = stored.get("note")
    if isinstance(note, str) and note.strip():
        lines.extend(["", note.strip()])
    if not worst:
        lines.extend(["", "There is insufficient evidence to list failure slices."])
        return lines
    overall = stored.get("overall_metrics") if isinstance(stored.get("overall_metrics"), dict) else {}
    rows = []
    for item in worst:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        rows.append(
            [
                item.get("feature"),
                item.get("slice_definition"),
                item.get("sample_count"),
                _metric(metrics, "f1"),
                _metric(overall, "f1"),
                _number(item.get("delta_f1")),
                _metric(metrics, "accuracy"),
                _metric(overall, "accuracy"),
                _number(item.get("delta_accuracy")),
            ]
        )
    lines.extend(
        [
            "",
            _table(
                [
                    "feature",
                    "slice",
                    "sample count",
                    "F1",
                    "overall F1",
                    "delta F1",
                    "accuracy",
                    "overall accuracy",
                    "delta accuracy",
                ],
                rows,
            ),
        ]
    )
    return lines


def _high_confidence_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 2. High-Confidence Errors",
        "",
        "These are the stored incorrect predictions with the highest confidence.",
        "",
    ]
    lines.extend(_model_blocks(payload, _high_confidence_block))
    return "\n".join(lines).rstrip()


def _high_confidence_block(run: dict[str, Any]) -> list[str]:
    component = _component(run, "hard_examples")
    lines = [_status_line(component)]
    stored = _stored(component)
    if stored is None:
        lines.append("There is insufficient evidence to list high-confidence errors.")
        return lines
    lines.extend(_count_lines(stored.get("summary")))
    examples = [
        item for item in stored.get("high_confidence_errors") or [] if isinstance(item, dict)
    ]
    if not examples:
        lines.extend(["", "No high-confidence errors were stored."])
        lines.append("There is insufficient evidence to list high-confidence errors.")
        return lines
    lines.extend(["", *_example_lines(examples, _experiment(run).get("class_labels"), "confidence")])
    return lines


def _uncertain_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 3. Most Uncertain Predictions",
        "",
        "These are the stored predictions with the highest uncertainty.",
        "",
    ]
    lines.extend(_model_blocks(payload, _uncertain_block))
    return "\n".join(lines).rstrip()


def _uncertain_block(run: dict[str, Any]) -> list[str]:
    component = _component(run, "hard_examples")
    lines = [_status_line(component)]
    stored = _stored(component)
    if stored is None:
        lines.append("There is insufficient evidence to list uncertain predictions.")
        return lines
    examples = [item for item in stored.get("most_uncertain") or [] if isinstance(item, dict)]
    if not examples:
        lines.extend(["", "No uncertain predictions were stored."])
        lines.append("There is insufficient evidence to list uncertain predictions.")
        return lines
    lines.extend(
        [
            "",
            *_example_lines(
                examples,
                _experiment(run).get("class_labels"),
                "probability",
                "uncertainty",
            ),
        ]
    )
    return lines


def _robustness_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 4. Robustness",
        "",
        "Noise level, mean F1, standard deviation, and performance drop are copied "
        "from the stored robustness results.",
        "",
    ]
    lines.extend(_model_blocks(payload, _robustness_block))
    comparison = _comparison_blocks(payload)
    if comparison:
        lines.extend(["", "### Comparison", "", *comparison])
    return "\n".join(lines).rstrip()


def _robustness_block(run: dict[str, Any]) -> list[str]:
    component = _component(run, "robustness")
    lines = [_status_line(component)]
    stored = _stored(component)
    levels = [] if stored is None else [item for item in stored.get("results") or [] if isinstance(item, dict)]
    if not levels:
        lines.append("There is insufficient evidence to report robustness.")
        return lines
    lines.extend(["", _robustness_table([( _experiment(run).get("model"), levels)])])
    return lines


def _comparison_blocks(payload: dict[str, Any]) -> list[str]:
    grouped: dict[Any, list[tuple[Any, list[dict[str, Any]]]]] = {}
    for run in _runs(payload):
        experiment = _experiment(run)
        stored = _stored(_component(run, "robustness"))
        if stored is None:
            continue
        levels = [item for item in stored.get("results") or [] if isinstance(item, dict)]
        if not levels:
            continue
        grouped.setdefault(experiment.get("dataset"), []).append((experiment.get("model"), levels))
    lines = []
    for dataset, models in grouped.items():
        if len(models) < 2:
            continue
        lines.extend(
            [
                f"Stored robustness results for `{dataset}` compared across models.",
                "",
                _robustness_table(models),
                "",
            ]
        )
    return lines


def _robustness_table(models: list[tuple[Any, list[dict[str, Any]]]]) -> str:
    levels = _ordered_levels(models)
    if len(models) == 1:
        _model_name, model_levels = models[0]
        lookup = _level_lookup(model_levels)
        rows = []
        for level in levels:
            item = lookup.get(level)
            mean, std, drop = _f1_trio(item)
            rows.append([_number(level), mean, std, drop])
        return _table(
            ["noise level", "mean F1", "standard deviation", "performance drop"],
            rows,
        )
    headers = ["noise level"]
    lookups = []
    for model_name, model_levels in models:
        headers.extend(
            [
                f"{model_name} mean F1",
                f"{model_name} standard deviation",
                f"{model_name} performance drop",
            ]
        )
        lookups.append(_level_lookup(model_levels))
    rows = []
    for level in levels:
        row = [_number(level)]
        for lookup in lookups:
            mean, std, drop = _f1_trio(lookup.get(level))
            row.extend([mean, std, drop])
        rows.append(row)
    return _table(headers, rows)


def _importance_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 5. Feature Importance",
        "",
        "Ranks and importance values are copied from the stored explanation.",
        "",
    ]
    lines.extend(_model_blocks(payload, _importance_block))
    return "\n".join(lines).rstrip()


def _importance_block(run: dict[str, Any]) -> list[str]:
    component = _component(run, "explanation")
    lines = [_status_line(component)]
    if component.get("status") == STATUS_NOT_SUPPORTED:
        reason = _reason(component) or "This model does not expose feature importances."
        lines.append(f"Feature importance is not supported. {reason}")
        lines.append("There is insufficient evidence to list ranked features.")
        return lines
    stored = _stored(component)
    features = [] if stored is None else [item for item in stored.get("features") or [] if isinstance(item, dict)]
    if not features:
        lines.append("There is insufficient evidence to list ranked features.")
        return lines
    rows = [
        [item.get("rank"), item.get("feature"), _number(item.get("importance"))]
        for item in features
    ]
    lines.extend(["", _table(["rank", "feature", "importance"], rows)])
    return lines


def _summary_section(payload: dict[str, Any]) -> str:
    lines = [
        "## 6. Diagnostic Summary",
        "",
        "These statements are filled from stored diagnostic fields. "
        "They do not show that a feature caused the model to fail.",
        "",
    ]
    lines.extend(_model_blocks(payload, _summary_block))
    return "\n".join(lines).rstrip()


def _summary_block(run: dict[str, Any]) -> list[str]:
    lines = ["- " + _slice_statement(run)]
    lines.append("- " + _degradation_statement(run))
    lines.append("- " + _error_statement(run))
    lines.append("- " + _uncertainty_statement(run))
    lines.append("- " + _robustness_statement(run))
    lines.append("- " + _importance_statement(run))
    associations = _association_statements(run)
    if associations:
        lines.append("- Stored importance and slice associations:")
        lines.extend(f"  - {statement}" for statement in associations)
    elif _component(run, "diagnostic_summary").get("status") == STATUS_NOT_SUPPORTED:
        lines.append(
            "- Feature-importance associations are not supported for this model. "
            "There is insufficient evidence to join importance with failure slices."
        )
    else:
        lines.append(
            "- There is insufficient evidence of a stored association between "
            "high importance and a lower-F1 slice."
        )
    return lines


def _slice_statement(run: dict[str, Any]) -> str:
    item = _worst_slice(run)
    if item is None:
        return "There is insufficient evidence to identify the largest failure slice."
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    overall = _overall_metrics(run)
    return (
        "The largest failure slice is "
        f"{item.get('feature')} {item.get('slice_definition')} "
        f"({item.get('sample_count')} samples), with F1 {_number(metrics.get('f1'))} "
        f"compared with overall F1 {_number(overall.get('f1'))} "
        f"(delta {_number(item.get('delta_f1'))})."
    )


def _degradation_statement(run: dict[str, Any]) -> str:
    item = _worst_slice(run)
    if item is None or not _is_number(item.get("delta_f1")):
        return "There is insufficient evidence to identify the largest performance degradation."
    return (
        "The largest performance degradation among valid slices is an F1 delta of "
        f"{_number(item.get('delta_f1'))} for {item.get('feature')} "
        f"{item.get('slice_definition')}."
    )


def _error_statement(run: dict[str, Any]) -> str:
    stored = _stored(_component(run, "hard_examples"))
    if stored is None:
        return "There is insufficient evidence to describe notable high-confidence errors."
    examples = [
        item for item in stored.get("high_confidence_errors") or [] if isinstance(item, dict)
    ]
    summary = stored.get("summary") if isinstance(stored.get("summary"), dict) else {}
    if not examples:
        recorded = summary.get("n_errors")
        if recorded == 0:
            return "The stored summary records no incorrect predictions, so there are no notable high-confidence errors."
        return "There is insufficient evidence to describe notable high-confidence errors."
    top = examples[0]
    labels = _experiment(run).get("class_labels")
    recorded = summary.get("wrong_confident")
    described = (
        f"sample {top.get('sample_id')}: "
        f"actual {_class_name(labels, top.get('y_true'))}, "
        f"predicted {_class_name(labels, top.get('y_pred'))}, "
        f"confidence {_number(top.get('confidence'))}"
    )
    if _is_confident_error(top, summary):
        count_text = (
            f"{recorded} incorrect and confident predictions are recorded. "
            if _is_number(recorded)
            else ""
        )
        return f"{count_text}A notable high-confidence error is {described}."
    if _is_number(recorded) and int(recorded) == 0:
        return (
            "The stored summary records no incorrect predictions at or above "
            "the confidence threshold. "
            f"The highest-confidence incorrect prediction is {described}."
        )
    return f"The highest-confidence incorrect prediction is {described}."


def _uncertainty_statement(run: dict[str, Any]) -> str:
    stored = _stored(_component(run, "hard_examples"))
    if stored is None:
        return "There is insufficient evidence to describe uncertainty patterns."
    summary = stored.get("summary") if isinstance(stored.get("summary"), dict) else {}
    examples = [item for item in stored.get("most_uncertain") or [] if isinstance(item, dict)]
    parts = []
    if _is_number(summary.get("correct_uncertain")) or _is_number(summary.get("wrong_uncertain")):
        parts.append(
            "The stored summary records "
            f"{summary.get('correct_uncertain', 'not available')} correct uncertain predictions and "
            f"{summary.get('wrong_uncertain', 'not available')} incorrect uncertain predictions."
        )
    if examples:
        top = examples[0]
        parts.append(
            f"The most uncertain stored prediction is sample {top.get('sample_id')}, "
            f"with probability {_number(top.get('probability'))} and "
            f"uncertainty {_number(top.get('uncertainty'))}."
        )
    if not parts:
        return "There is insufficient evidence to describe uncertainty patterns."
    return " ".join(parts)


def _robustness_statement(run: dict[str, Any]) -> str:
    stored = _stored(_component(run, "robustness"))
    if stored is None:
        return "There is insufficient evidence to describe robustness degradation."
    chosen = _largest_f1_drop(stored.get("results") or [])
    if chosen is None:
        return "There is insufficient evidence to describe robustness degradation."
    drop = chosen["degradation"]["f1"]
    level = chosen.get("noise_level")
    mean = chosen.get("mean", {}).get("f1")
    std = chosen.get("std", {}).get("f1")
    if float(drop) > 0:
        change = f"The largest robustness degradation is an F1 drop of {_number(drop)}"
    elif float(drop) == 0:
        return (
            "Stored robustness results show no F1 degradation at the recorded noise levels. "
            f"At noise level {_number(level)}, mean F1 is {_number(mean)} "
            f"and the standard deviation is {_number(std)}."
        )
    else:
        change = f"Stored robustness results do not show an F1 drop. The recorded F1 change is {_number(drop)}"
    return (
        f"{change} at noise level {_number(level)}. "
        f"Mean F1 is {_number(mean)} and the standard deviation is {_number(std)}."
    )


def _importance_statement(run: dict[str, Any]) -> str:
    component = _component(run, "explanation")
    if component.get("status") == STATUS_NOT_SUPPORTED:
        reason = _reason(component) or "This model does not expose feature importances."
        return (
            f"Feature importance is not supported. {reason} "
            "There is insufficient evidence to describe important features."
        )
    stored = _stored(component)
    features = [] if stored is None else [item for item in stored.get("features") or [] if isinstance(item, dict)]
    if not features:
        return "There is insufficient evidence to describe important features."
    top = features[0]
    return (
        f"The highest ranked important feature is {top.get('feature')} "
        f"(rank {top.get('rank')}, importance {_number(top.get('importance'))})."
    )


def _association_statements(run: dict[str, Any]) -> list[str]:
    stored = _stored(_component(run, "diagnostic_summary"))
    if stored is None:
        return []
    statements = []
    for item in stored.get("associations") or []:
        if isinstance(item, dict) and isinstance(item.get("statement"), str):
            statements.append(item["statement"])
    return statements


def _worst_slice(run: dict[str, Any]) -> dict[str, Any] | None:
    stored = _stored(_component(run, "slicing"))
    if stored is None:
        return None
    worst = [item for item in stored.get("worst_slices") or [] if isinstance(item, dict)]
    if not worst:
        return None
    return worst[0]


def _overall_metrics(run: dict[str, Any]) -> dict[str, Any]:
    stored = _stored(_component(run, "slicing"))
    if stored is None or not isinstance(stored.get("overall_metrics"), dict):
        return {}
    return stored["overall_metrics"]


def _largest_f1_drop(levels) -> dict[str, Any] | None:
    chosen = None
    chosen_drop = None
    for level in levels:
        if not isinstance(level, dict):
            continue
        degradation = level.get("degradation")
        if not isinstance(degradation, dict) or not _is_number(degradation.get("f1")):
            continue
        drop = float(degradation["f1"])
        if chosen is None or drop > chosen_drop:
            chosen = level
            chosen_drop = drop
    return chosen


def _example_lines(examples, class_labels, *value_fields: str) -> list[str]:
    headers = ["sample ID", "actual", "predicted", *[_field_label(field) for field in value_fields]]
    rows = []
    for example in examples:
        rows.append(
            [
                example.get("sample_id"),
                _class_name(class_labels, example.get("y_true")),
                _class_name(class_labels, example.get("y_pred")),
                *[_number(example.get(field)) for field in value_fields],
            ]
        )
    lines = [_table(headers, rows)]
    for example in examples:
        lines.extend(["", f"Sample {example.get('sample_id')} feature values:", ""])
        features = example.get("features")
        if isinstance(features, dict) and features:
            lines.append(_table(["feature", "value"], [[name, value] for name, value in features.items()]))
        else:
            lines.append("No feature values were stored for this example.")
    return lines


def _count_lines(summary) -> list[str]:
    if not isinstance(summary, dict):
        return []
    rows = []
    for key, label in _SUMMARY_COUNTS:
        if key in summary:
            rows.append([label, summary.get(key)])
    threshold = summary.get("confidence_threshold")
    if _is_number(threshold):
        rows.append(["confidence threshold", _number(threshold)])
    if not rows:
        return []
    return ["", _table(["group", "stored count"], rows)]


def _model_blocks(payload: dict[str, Any], builder) -> list[str]:
    runs = _runs(payload)
    if not runs:
        return ["There is insufficient evidence to report this section."]
    lines = []
    for run in runs:
        if lines:
            lines.append("")
        lines.extend([f"### {_model_heading(_experiment(run))}", "", *builder(run)])
    return lines


def _status_line(component: dict[str, Any]) -> str:
    status = component.get("status")
    detail = component.get("detail")
    if status == STATUS_SUPPORTED and not detail:
        return f"Status: {status}."
    if detail:
        return f"Status: {status}. {detail}"
    return f"Status: {status}."


def _reason(component: dict[str, Any]) -> str | None:
    stored = _stored(component)
    if stored is not None and isinstance(stored.get("reason"), str) and stored["reason"].strip():
        return stored["reason"].strip()
    detail = component.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return None


def _f1_trio(level: dict[str, Any] | None) -> tuple[str, str, str]:
    if not isinstance(level, dict):
        return ("not available", "not available", "not available")
    mean = level.get("mean") if isinstance(level.get("mean"), dict) else {}
    std = level.get("std") if isinstance(level.get("std"), dict) else {}
    degradation = level.get("degradation") if isinstance(level.get("degradation"), dict) else {}
    return (_number(mean.get("f1")), _number(std.get("f1")), _number(degradation.get("f1")))


def _level_lookup(levels: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {level.get("noise_level"): level for level in levels}


def _ordered_levels(models: list[tuple[Any, list[dict[str, Any]]]]) -> list[Any]:
    ordered = []
    for _model, levels in models:
        for level in levels:
            value = level.get("noise_level")
            if value not in ordered:
                ordered.append(value)
    return ordered


def _component(run: dict[str, Any], name: str) -> dict[str, Any]:
    value = run.get(name)
    if isinstance(value, dict):
        return value
    return {
        "name": name,
        "status": STATUS_INSUFFICIENT_DATA,
        "detail": "This component was not included in the diagnostic result.",
        "result": None,
    }


def _stored(component: dict[str, Any]) -> dict[str, Any] | None:
    result = component.get("result")
    if isinstance(result, dict):
        return result
    return None


def _experiment(run: dict[str, Any]) -> dict[str, Any]:
    experiment = run.get("experiment")
    if isinstance(experiment, dict):
        return experiment
    return {}


def _runs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [run for run in payload.get("runs") or [] if isinstance(run, dict)]


def _payload(result: DiagnosticEngineResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, DiagnosticEngineResult):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        raise TypeError(
            "The report generator expects a DiagnosticEngineResult or its dictionary."
        )
    if "runs" not in payload:
        raise ValueError("The diagnostic result has no runs.")
    if not isinstance(payload["runs"], list):
        raise TypeError("Diagnostic runs must be a list.")
    return payload


def _destination(path: Path | str | None, directory: Path | str | None, default: Path) -> Path:
    if path is not None:
        return Path(path)
    if directory is not None:
        return Path(directory) / default.name
    return default


def _table(headers: list[Any], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(_cell(header) for header in headers) + " |"
    rule = "| " + " | ".join("---" for _header in headers) + " |"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _cell(value: Any) -> str:
    if _is_number(value):
        text = _number(value)
    elif value is None:
        text = "not available"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _metric(metrics: dict[str, Any], key: str) -> str:
    if key not in metrics:
        return "not available"
    return _number(metrics.get(key))


def _number(value: Any) -> str:
    if not _is_number(value):
        return "not available"
    return f"{float(value):.{_DISPLAY_DECIMALS}f}"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_confident_error(example: dict[str, Any], summary: dict[str, Any]) -> bool:
    """Return whether this stored error meets the confidence threshold."""
    category = example.get("category")
    if category == "wrong_confident":
        return True
    if category == "wrong_uncertain":
        return False
    threshold = summary.get("confidence_threshold")
    confidence = example.get("confidence")
    if not _is_number(threshold) or not _is_number(confidence):
        return False
    return float(confidence) >= float(threshold)


def _class_name(class_labels: Any, value: Any) -> str:
    if not isinstance(class_labels, dict):
        return "not available" if value is None else str(value)
    for key, name in class_labels.items():
        if key == value or str(key) == str(value):
            return f"{value} ({name})"
        if _is_number(key) and _is_number(value) and int(key) == int(value):
            return f"{value} ({name})"
    return "not available" if value is None else str(value)


def _class_label_text(class_labels: dict[Any, Any]) -> str:
    def sort_key(key: Any):
        if _is_number(key) or (isinstance(key, str) and key.lstrip("-").isdigit()):
            return (0, int(key))
        return (1, str(key))

    return ", ".join(f"{key} = {class_labels[key]}" for key in sorted(class_labels, key=sort_key))


def _joined(values: Any) -> str:
    if isinstance(values, (list, tuple)) and values:
        return ", ".join(str(value) for value in values)
    return "not available"


def _model_heading(experiment: dict[str, Any]) -> str:
    dataset = experiment.get("dataset") or "unknown dataset"
    model = experiment.get("model") or "unknown model"
    return f"{dataset} / {model}"


def _field_label(field: str) -> str:
    labels = {
        "confidence": "confidence",
        "probability": "probability",
        "uncertainty": "uncertainty",
    }
    return labels.get(field, field)
