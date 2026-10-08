"""Run the Phase 3 diagnostic engine for one configured experiment.

Uses the Phase 2 configuration and registry. Saved models and predictions
are reused. The model is trained only when that experiment has no artifacts
yet.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from src.diagnostics.diagnostic_engine import format_diagnostic_engine
from src.experiments.config import apply_overrides, load_config
from src.experiments.registry import list_datasets, list_models
from src.run_phase2 import format_summary, run_configured_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the same experiment selectors the Phase 2 command accepts."""
    datasets = ", ".join(list_datasets())
    models = ", ".join(list_models())
    parser = argparse.ArgumentParser(
        description=(
            "Run slicing, hard-example, robustness, and explanation "
            "diagnostics, then write the failure report. "
            "Reuses a saved experiment when it exists. "
            f"Available datasets: {datasets}."
        )
    )
    parser.add_argument(
        "--dataset",
        help=f"Dataset name. Overrides config.json. Available datasets: {datasets}.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model names, or 'all' for every registered model. "
            f"Overrides config.json. Available models: {models}."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Split and model seed. Overrides config.json.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of rows held out for evaluation. Overrides config.json.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration file. Defaults to config.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load config, apply overrides, and diagnose the selected experiment."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        config = apply_overrides(
            config,
            dataset=args.dataset,
            models=args.models,
            random_state=args.random_state,
            test_size=args.test_size,
        )
        config = replace(config, diagnostics=replace(config.diagnostics, enabled=True))
        outcome = run_configured_experiment(config)
    except (ValueError, TypeError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(outcome))
    if outcome.diagnostics is not None:
        print(format_diagnostic_engine(outcome.diagnostics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
