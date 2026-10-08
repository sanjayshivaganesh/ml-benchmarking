"""Run Phase 3 robustness testing for one configured experiment.

Uses the Phase 2 configuration and registry. A saved model is loaded and
scored on noisy copies of the test split. The model is trained only when
that experiment has no model artifact yet.
"""

from __future__ import annotations

import argparse
import sys

from src.diagnostics.workflow import format_robustness_summary, run_diagnostic_robustness
from src.experiments.config import apply_overrides, load_config
from src.experiments.registry import list_datasets, list_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the same experiment selectors the Phase 2 command accepts."""
    datasets = ", ".join(list_datasets())
    models = ", ".join(list_models())
    parser = argparse.ArgumentParser(
        description=(
            "Score saved models on numerical noise. Reuses a saved model "
            f"when it exists. Available datasets: {datasets}."
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
    """Load config, apply overrides, and test robustness for the experiment."""
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
        outcome = run_diagnostic_robustness(config)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_robustness_summary(outcome))
    return 0


if __name__ == "__main__":
    sys.exit(main())
