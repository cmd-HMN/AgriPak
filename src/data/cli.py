from __future__ import annotations

import argparse
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
    prog="python -m src.data",
    description="AgriPak Exploratory Data Analysis & Reporting Pipeline",
    default_input="data/raw",
    default_output="src/data/cleaned_data.csv",
    default_log_dir="log",
)

_SUBCOMMANDS = ("run", "audit", "clean", "analyze", "plot", "report", "help")
_DEFAULT_REPORT = Path("src/data/README.md")
_DEFAULT_FRAGMENT = Path("src/data/report_fragment.md")
_DEFAULT_PREPROCESS_FRAGMENT = Path("src/preprocess/report_fragment.md")


def _configure_extra(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=_SUBCOMMANDS,
    )
    parser.add_argument("--assets", type=Path, default=Path("src/data/assets"))
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--fragment", type=Path, default=_DEFAULT_FRAGMENT)
    parser.add_argument(
        "--preprocess-fragment", type=Path, default=_DEFAULT_PREPROCESS_FRAGMENT
    )
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--leak-threshold", type=float, default=0.90)


def _resolve_csv_output(output: Path) -> Path:
    return output if output.suffix else output / "cleaned_data.csv"


def _cmd_audit(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.data.analysis import audit_data_quality

        tick("Auditing dataset")
        raw_df = load_dataset(args.input)
        state["rows"] = len(raw_df)
        state["cols"] = len(raw_df.columns)
        state["qa"] = audit_data_quality(raw_df)

    run_with_progress(console, 2, "Auditing Data", _run)
    qa = state["qa"]
    logger.info("Audit complete: %d rows, %d cols", state["rows"], state["cols"])
    print_summary_table(
        console,
        "Data Audit",
        [
            ("Rows", f"{state['rows']:,}"),
            ("Columns", str(state["cols"])),
            ("Duplicates", str(qa.duplicate_count)),
        ],
    )
    return EXIT_SUCCESS


def _cmd_clean(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    out_path = _resolve_csv_output(args.output)
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset, save_dataframe
        from src.data.cleaning import clean_dataset

        tick("Loading dataset")
        raw_df = load_dataset(args.input)
        res = clean_dataset(raw_df, on_step=tick)
        tick("Saving cleaned CSV")
        save_dataframe(res.data, out_path, source_input=args.input)
        state["res"] = res

    run_with_progress(console, 8, "Cleaning Data", _run)
    res = state["res"]
    logger.info("Cleaned data written to %s (%d rows)", out_path, res.rows_after)
    print_summary_table(
        console,
        "Data Cleaning",
        [
            ("Rows Before", f"{res.rows_before:,}"),
            ("Rows After", f"{res.rows_after:,}"),
            ("Output", str(out_path)),
        ],
    )
    return EXIT_SUCCESS


def _cmd_analyze(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.data.analysis import run_full_analysis
        from src.data.cleaning import clean_dataset

        tick("Loading dataset")
        raw_df = load_dataset(args.input)
        clean_res = clean_dataset(raw_df, on_step=tick)
        tick("Running statistical analysis")
        state["bundle"] = run_full_analysis(
            clean_res.data,
            clean_res.schema,
            raw_text_df=raw_df,
            min_n=args.min_n,
            n_splits=args.n_splits,
            leak_threshold=args.leak_threshold,
        )

    run_with_progress(console, 8, "Analyzing Data", _run)
    bundle = state["bundle"]
    logger.info("Analysis complete: %d rows", bundle.row_count)
    print_summary_table(
        console,
        "Data Analysis",
        [
            ("Rows", f"{bundle.row_count:,}"),
            ("Columns", str(bundle.col_count)),
            ("Leakage Flags", str(len(bundle.flagged_leakage_columns))),
        ],
    )
    return EXIT_SUCCESS


def _cmd_plot(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.data.analysis import run_full_analysis
        from src.data.cleaning import clean_dataset
        from src.data.plotting import generate_all_plots

        tick("Loading dataset")
        raw_df = load_dataset(args.input)
        clean_res = clean_dataset(raw_df, on_step=tick)
        tick("Running analysis")
        bundle = run_full_analysis(
            clean_res.data,
            clean_res.schema,
            raw_text_df=raw_df,
            min_n=args.min_n,
            n_splits=args.n_splits,
            leak_threshold=args.leak_threshold,
        )
        tick("Generating plots")
        state["plots"] = generate_all_plots(clean_res.data, bundle, args.assets)

    run_with_progress(console, 9, "Generating Plots", _run)
    plots = state["plots"]
    generated = sum(1 for p in plots if p.generated)
    logger.info("Generated %d/%d plots in %s", generated, len(plots), args.assets)
    print_summary_table(
        console,
        "Data Plots",
        [
            ("Generated", f"{generated}/{len(plots)}"),
            ("Assets Dir", str(args.assets)),
        ],
    )
    return EXIT_SUCCESS


def _cmd_report(
    args: argparse.Namespace, console: Console, logger: logging.Logger
) -> int:
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset
        from src.data.analysis import run_full_analysis
        from src.data.cleaning import clean_dataset
        from src.data.plotting import generate_all_plots
        from src.data.report import generate_report

        tick("Loading dataset")
        raw_df = load_dataset(args.input)
        clean_res = clean_dataset(raw_df, on_step=tick)
        tick("Running analysis")
        bundle = run_full_analysis(
            clean_res.data,
            clean_res.schema,
            raw_text_df=raw_df,
            min_n=args.min_n,
            n_splits=args.n_splits,
            leak_threshold=args.leak_threshold,
        )
        tick("Generating plots")
        plots = generate_all_plots(clean_res.data, bundle, args.assets)
        tick("Writing report")
        state["out"] = generate_report(
            raw_df,
            clean_res,
            bundle,
            plots,
            args.input,
            report_path=args.report,
            fragment_path=args.fragment,
            preprocess_fragment_path=args.preprocess_fragment,
        )
        state["plots"] = plots
        state["bundle"] = bundle

    run_with_progress(console, 10, "Data Report", _run)
    out = state["out"]
    plots = state["plots"]
    bundle = state["bundle"]
    generated = sum(1 for p in plots if p.generated)
    logger.info("Report written to %s", out)
    print_summary_table(
        console,
        "Data Report",
        [
            ("Rows", f"{bundle.row_count:,}"),
            ("Plots", f"{generated}/{len(plots)}"),
            ("Report", str(out)),
        ],
    )
    return EXIT_SUCCESS


def _cmd_run(args: argparse.Namespace, console: Console, logger: logging.Logger) -> int:
    out_csv = _resolve_csv_output(args.output)
    state: dict[str, Any] = {}

    def _run(tick: Any) -> None:
        tick("Loading modules")
        from src.common.core.io import load_dataset, save_dataframe
        from src.data.analysis import run_full_analysis
        from src.data.cleaning import clean_dataset
        from src.data.plotting import generate_all_plots
        from src.data.report import generate_report

        tick("Loading dataset")
        raw_df = load_dataset(args.input)
        clean_res = clean_dataset(raw_df, on_step=tick)
        tick("Saving cleaned CSV")
        save_dataframe(clean_res.data, out_csv, source_input=args.input)
        tick("Running analysis")
        bundle = run_full_analysis(
            clean_res.data,
            clean_res.schema,
            raw_text_df=raw_df,
            min_n=args.min_n,
            n_splits=args.n_splits,
            leak_threshold=args.leak_threshold,
        )
        tick("Generating plots")
        plots = generate_all_plots(clean_res.data, bundle, args.assets)
        tick("Writing report")
        out_report = generate_report(
            raw_df,
            clean_res,
            bundle,
            plots,
            args.input,
            report_path=args.report,
            fragment_path=args.fragment,
            preprocess_fragment_path=args.preprocess_fragment,
        )
        state["bundle"] = bundle
        state["plots"] = plots
        state["out_report"] = out_report

    run_with_progress(console, 11, "Data Pipeline", _run)
    bundle = state["bundle"]
    plots = state["plots"]
    generated = sum(1 for p in plots if p.generated)
    logger.info("Pipeline complete: %s", state["out_report"])
    print_summary_table(
        console,
        "Data Pipeline",
        [
            ("Rows", f"{bundle.row_count:,}"),
            ("Plots", f"{generated}/{len(plots)}"),
            ("Cleaned CSV", str(out_csv)),
            ("Report", str(state["out_report"])),
        ],
    )
    return EXIT_SUCCESS


def _execute(args: argparse.Namespace, console: Console, logger: logging.Logger) -> int:
    if args.command == "help":
        print_cli_help(build_cli_parser(_SPEC, _configure_extra), console)
        return EXIT_SUCCESS
    handlers: dict[str, Any] = {
        "run": _cmd_run,
        "audit": _cmd_audit,
        "clean": _cmd_clean,
        "analyze": _cmd_analyze,
        "plot": _cmd_plot,
        "report": _cmd_report,
    }
    fn = handlers.get(args.command)
    if fn is None:
        raise InvalidArgumentError(f"Unknown command: {args.command}")
    return int(fn(args, console, logger))


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli_app(
        _SPEC,
        argv,
        _execute,
        configure_extra=_configure_extra,
        logger_name="agripak.data",
    )
