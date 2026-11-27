from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from src.common.cli import (
    CLISpec,
    build_cli_parser,
    print_cli_help,
    print_summary_table,
    run_cli_app,
    run_with_progress,
)
from src.common.core.errors import EXIT_SUCCESS, InvalidArgumentError

_SPEC = CLISpec(
    prog="python -m src.model",
    description="AgriPak Baseline Model Training & Evaluation CLI",
    default_input="data/splits",
    default_output=None,
    default_log_dir="log",
)

_SUBCOMMANDS = ("baseline", "help")


def _configure_extra(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "command",
        nargs="?",
        default="baseline",
        choices=_SUBCOMMANDS,
    )
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", "--output-dir", dest="output", type=Path, default=None
    )


def _cmd_baseline(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules", 0.0)
        from src.model.baseline.logistic import (
            load_split_folder,
            train_evaluate_logistic,
        )

        tick("Loading train/test splits", 20.0)
        train_df, test_df = load_split_folder(Path(args.input))
        tick("Training Logistic Regression baseline", 55.0)
        metrics = train_evaluate_logistic(
            train_df,
            test_df,
            target_column=args.target,
            random_state=args.seed,
        )
        state["metrics"] = metrics
        if args.output is not None:
            tick("Saving metrics JSON", 15.0)
            from src.common.core.io import write_text_file

            out_path = (
                Path(args.output)
                if Path(args.output).suffix
                else Path(args.output) / "baseline_metrics.json"
            )
            payload = {
                "target_column": metrics.target_column,
                "train_rows": metrics.train_rows,
                "test_rows": metrics.test_rows,
                "n_features": metrics.n_features,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "confusion_matrix": [list(r) for r in metrics.confusion_matrix],
            }
            state["out_path"] = write_text_file(
                json.dumps(payload, indent=2) + "\n", out_path
            )

    run_with_progress(console, 100.0, "Logistic Regression Baseline", _run)
    m = state["metrics"]
    logger.info(
        "Baseline test metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f",
        m.accuracy,
        m.precision,
        m.recall,
        m.f1,
        m.roc_auc,
    )
    rows = [
        ("Target", m.target_column),
        ("Train / Test Rows", f"{m.train_rows:,} / {m.test_rows:,}"),
        ("Features Used", str(m.n_features)),
        ("Accuracy", f"{m.accuracy:.4f}"),
        ("Precision", f"{m.precision:.4f}"),
        ("Recall", f"{m.recall:.4f}"),
        ("F1 Score", f"{m.f1:.4f}"),
        ("ROC-AUC", f"{m.roc_auc:.4f}"),
    ]
    if "out_path" in state:
        rows.append(("Metrics JSON", str(state["out_path"])))
    print_summary_table(console, "Logistic Regression Baseline (Test Set)", rows)
    return EXIT_SUCCESS


def _execute(args: argparse.Namespace, console: Console, logger: logging.Logger) -> int:
    if args.command == "help":
        print_cli_help(build_cli_parser(_SPEC, _configure_extra), console)
        return EXIT_SUCCESS
    if args.command == "baseline":
        return _cmd_baseline(args, console, logger)
    raise InvalidArgumentError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli_app(
        _SPEC,
        argv,
        _execute,
        configure_extra=_configure_extra,
        logger_name="agripak.model",
    )
