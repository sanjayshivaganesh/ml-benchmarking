"""Command-line entry for one Phase 2 experiment.

Argument parsing, configuration overrides, and validation happen here.
Dataset loading, training, metrics, and error analysis stay in the Phase 1
modules reached through ``run_experiment``.
"""

from __future__ import annotations

import argparse
import sys

from src.experiments.config import apply_overrides, load_config
from src.experiments.registry import list_datasets, list_models
from src.run_phase2 import format_summary, run_phase2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line overrides for ``config.json``."""
    datasets = ", ".join(list_datasets())
    models = ", ".join(list_models())
    parser = argparse.ArgumentParser(
        description="Run one experiment on a Phase 1 dataset and selected models."
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
    """Load config, apply overrides, run one experiment, and print a summary."""
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
        outcome = run_phase2(config)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(outcome))
    return 0


if __name__ == "__main__":
    sys.exit(main())
