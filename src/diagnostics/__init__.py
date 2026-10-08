"""Shared inputs for model failure diagnostics.

Slicing and hard-example analysis read a ``PredictionStore`` instead of
calling ``predict``. Robustness testing calls ``predict`` on copies of the
test features. Feature importance reads a fitted pipeline and does not fit it.
Importance and slices can be joined into an association summary. That
summary does not treat a feature as the cause of an error.
"""

from .diagnostic_summary import (
    DEFAULT_SUBSTANTIAL_F1_DROP,
    DEFAULT_TOP_FEATURES,
    DiagnosticSummary,
    build_diagnostic_summary,
)
from .explain import (
    FeatureImportanceResult,
    explain_feature_importance,
    save_feature_importance,
)
from .hard_examples import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_N,
    HardExampleResult,
    explore_hard_examples,
    save_experiment_hard_examples,
)
from .robustness import (
    DEFAULT_N_SEEDS,
    DEFAULT_NOISE_LEVELS,
    RobustnessResult,
    compare_robustness,
    compare_robustness_files,
    perturb_features,
    run_robustness,
)
from .prediction_store import (
    PREDICTION_COLUMNS,
    PredictionStore,
    prediction_directory,
    prediction_filename,
    save_experiment_predictions,
)
from .slicing import (
    DEFAULT_NUMERIC_BINS,
    MIN_SLICE_SIZE,
    ErrorSlicingResult,
    run_error_slicing,
    save_experiment_slices,
)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_N_SEEDS",
    "DEFAULT_NOISE_LEVELS",
    "DEFAULT_NUMERIC_BINS",
    "DEFAULT_SUBSTANTIAL_F1_DROP",
    "DEFAULT_TOP_FEATURES",
    "DEFAULT_TOP_N",
    "MIN_SLICE_SIZE",
    "PREDICTION_COLUMNS",
    "DiagnosticSummary",
    "ErrorSlicingResult",
    "FeatureImportanceResult",
    "HardExampleResult",
    "PredictionStore",
    "RobustnessResult",
    "build_diagnostic_summary",
    "compare_robustness",
    "compare_robustness_files",
    "explain_feature_importance",
    "explore_hard_examples",
    "perturb_features",
    "prediction_directory",
    "prediction_filename",
    "run_error_slicing",
    "run_robustness",
    "save_experiment_hard_examples",
    "save_feature_importance",
    "save_experiment_predictions",
    "save_experiment_slices",
]
