"""Single-dataset experiments on top of the Phase 1 pipeline."""

from .config import (
    DEFAULT_CONFIG_PATH,
    ExperimentConfig,
    apply_overrides,
    load_config,
)
from .experiment import (
    DEFAULT_EXPERIMENT_DIR,
    ExperimentResult,
    ModelResult,
    run_experiment,
)
from .registry import (
    list_datasets,
    list_models,
    resolve_models,
    validate_dataset,
    validate_models,
)
from .visualization import (
    metric_comparison_table,
    plot_confusion_matrices,
    plot_confusion_matrix,
    plot_roc_curves,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXPERIMENT_DIR",
    "ExperimentConfig",
    "ExperimentResult",
    "ModelResult",
    "apply_overrides",
    "list_datasets",
    "list_models",
    "load_config",
    "metric_comparison_table",
    "plot_confusion_matrices",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "resolve_models",
    "run_experiment",
    "validate_dataset",
    "validate_models",
]
