from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.data.errors import (
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    DataPipelineError,
    InvalidArgumentError,
    ReportWriteError,
    format_cli_error,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.data.analysis import AnalysisBundle
    from src.data.cleaning import CleaningResult
    from src.data.plotting import PlotArtifact

audit_data_quality: Any = None
run_full_analysis: Any = None


class _CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise InvalidArgumentError(message)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("src/data"))
    parser.add_argument("--assets", type=Path, default=Path("src/assets"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--leak-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")


_COMMAND_HELP: tuple[tuple[str, str], ...] = (
    ("run", "Run full pipeline (clean, analyze, features, plots, report)"),
    ("audit", "Infer schema and run data quality and text-hygiene audit"),
    ("clean", "Run 5-step cleaning pipeline and save cleaned CSV + audit"),
    ("analyze", "Run statistical EDA, leakage check, and CV benchmarks"),
    ("features", "Engineer plot-driven features and export CSV"),
    ("plot", "Generate all 15 exploratory and engineered PNG charts"),
    ("report", "Generate plots and write src/data/README.md"),
    ("help", "Show available commands or help for a specific command"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = _CLIArgumentParser(prog="src.data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "audit", "clean", "analyze", "features", "plot", "report"):
        _add_common_arguments(subparsers.add_parser(name))
    help_parser = subparsers.add_parser("help")
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=("run", "audit", "clean", "analyze", "features", "plot", "report"),
        default=None,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise InvalidArgumentError(f"--n-splits must be >= 2, got {args.n_splits}")
    if args.min_n is not None and args.min_n < 1:
        raise InvalidArgumentError(f"--min-n must be >= 1, got {args.min_n}")
    if not (0.0 < args.leak_threshold <= 1.0):
        raise InvalidArgumentError(
            f"--leak-threshold must be in (0, 1], got {args.leak_threshold}"
        )


_pending_summary: tuple[str, list[tuple[str, str]]] | None = None


def _print_summary(title: str, rows: Sequence[tuple[str, str]]) -> None:
    global _pending_summary
    if _active_step is not None:
        _pending_summary = (title, list(rows))
        return
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for key, val in rows:
        table.add_row(key, val)
    Console().print(table)


def _write_markdown_file(path: Path, content: str) -> Path:
    from src.data.utils import ensure_directory

    target = Path(path)
    ensure_directory(target.parent)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(f"Failed to write '{target}': {exc}") from exc
    return target


_COMMAND_STEPS: dict[str, int] = {
    "audit": 4,
    "clean": 7,
    "features": 8,
    "analyze": 19,
    "plot": 33,
    "report": 34,
    "run": 35,
}

_active_step: Callable[[str], None] | None = None


def _step(label: str) -> None:
    if _active_step is not None:
        _active_step(label)


def _generate_plots_with_eta(
    clean_df: pd.DataFrame, bundle: AnalysisBundle, assets_dir: Path
) -> list[PlotArtifact]:
    from src.data.plotting import generate_all_plots

    return generate_all_plots(
        clean_df,
        bundle,
        assets_dir,
        on_plot=lambda name: _step(f"Plotting {name}..."),
    )


def _load_clean_analyze(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, CleaningResult, AnalysisBundle]:
    from src.data.analysis import run_full_analysis as default_run_full_analysis
    from src.data.cleaning import clean_dataset
    from src.data.loading import load_dataset, load_raw_text

    _step("Loading dataset...")
    raw_df = load_dataset(args.input)
    raw_text_df = load_raw_text(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target, on_step=_step)
    analyzer = (
        run_full_analysis
        if run_full_analysis is not None
        else default_run_full_analysis
    )
    bundle = analyzer(
        cleaning_result.data,
        cleaning_result.schema,
        raw_text_df=raw_text_df,
        min_n=args.min_n,
        leak_threshold=args.leak_threshold,
        n_splits=args.n_splits,
        random_state=args.seed,
        on_step=_step,
    )
    return raw_df, cleaning_result, bundle


def _render_report_bundle(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    bundle: AnalysisBundle,
    args: argparse.Namespace,
) -> None:
    from src.data.report import generate_report

    plots = _generate_plots_with_eta(cleaning_result.data, bundle, args.assets)
    _step("Writing report...")
    generate_report(
        raw_df=raw_df,
        cleaning_result=cleaning_result,
        analysis=bundle,
        plots=plots,
        input_path=args.input,
        report_path=args.output / "README.md",
    )


def _run_audit(args: argparse.Namespace) -> int:
    from src.data.analysis import audit_data_quality as default_audit_data_quality
    from src.data.loading import load_dataset, load_raw_text
    from src.data.schema import infer_schema
    from src.data.utils import dataframe_to_markdown

    _step("Loading dataset...")
    raw_df = load_dataset(args.input)
    raw_text_df = load_raw_text(args.input)
    _step("Inferring schema...")
    schema = infer_schema(raw_df, target_column=args.target)
    _step("Auditing data quality...")
    auditor = (
        audit_data_quality
        if audit_data_quality is not None
        else default_audit_data_quality
    )
    qa = auditor(raw_df, raw_text_df=raw_text_df)
    lines = [
        "# Data Quality Audit",
        "",
        f"- **Rows:** `{len(raw_df):,}`",
        f"- **Columns:** `{len(raw_df.columns)}`",
        f"- **Exact duplicates:** `{qa.duplicate_count}`",
        f"- **Memory (MB):** `{qa.memory_mb:.2f}`",
        f"- **Inferred target:** `{schema.target_column}`",
        "",
        "## Column Summary",
        "",
        dataframe_to_markdown(qa.column_summary, float_digits=3),
        "",
        "## Text Hygiene",
        "",
        dataframe_to_markdown(qa.text_hygiene, float_digits=0),
        "",
    ]
    _step("Writing quality audit...")
    out_path = _write_markdown_file(args.output / "quality_audit.md", "\n".join(lines))
    logging.getLogger("agripak.data").info("Wrote quality audit to %s", out_path)
    _print_summary(
        "Data Quality Audit",
        [
            ("Rows", f"{len(raw_df):,}"),
            ("Columns", str(len(raw_df.columns))),
            ("Duplicates", str(qa.duplicate_count)),
            ("Inferred target", schema.target_column),
            ("Output", str(out_path)),
        ],
    )
    return EXIT_SUCCESS


def _run_clean(args: argparse.Namespace) -> int:
    from src.data.cleaning import clean_dataset
    from src.data.loading import load_dataset, save_dataframe

    _step("Loading dataset...")
    raw_df = load_dataset(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target, on_step=_step)
    _step("Saving cleaned outputs...")
    csv_path = save_dataframe(cleaning_result.data, args.output / "cleaned_data.csv")
    lines = [
        "# Data Cleaning Audit",
        "",
        f"- **Rows before:** `{cleaning_result.rows_before:,}`",
        f"- **Rows after:** `{cleaning_result.rows_after:,}`",
        f"- **Memory before (MB):** `{cleaning_result.memory_mb_before:.2f}`",
        f"- **Memory after (MB):** `{cleaning_result.memory_mb_after:.2f}`",
        "",
        cleaning_result.audit.to_markdown(),
        "",
    ]
    md_path = _write_markdown_file(args.output / "cleaning_audit.md", "\n".join(lines))
    logging.getLogger("agripak.data").info("Wrote cleaned data to %s", csv_path)
    _print_summary(
        "Data Cleaning",
        [
            (
                "Rows",
                f"{cleaning_result.rows_before:,} -> {cleaning_result.rows_after:,}",
            ),
            (
                "Memory",
                f"{cleaning_result.memory_mb_before:.2f} MB -> {cleaning_result.memory_mb_after:.2f} MB",
            ),
            ("Cleaned CSV", str(csv_path)),
            ("Audit", str(md_path)),
        ],
    )
    return EXIT_SUCCESS


def _run_analyze(args: argparse.Namespace) -> int:
    from src.data.utils import dataframe_to_markdown

    _, _, bundle = _load_clean_analyze(args)
    _step("Writing analysis summary...")
    lines = [
        "# Analysis Summary",
        "",
        f"- **Rows:** `{bundle.row_count:,}`",
        f"- **Columns:** `{bundle.col_count}`",
        f"- **Min subgroup size (`min_n`):** `{bundle.min_n}`",
        f"- **Flagged leakage columns:** `{len(bundle.flagged_leakage_columns)}`",
        "",
        "## Ranked Target Spearman Correlations",
        "",
        dataframe_to_markdown(bundle.ranked_target_spearman, float_digits=4),
        "",
        "## Cross-Validated ROC-AUC",
        "",
        dataframe_to_markdown(bundle.cv_auc_table, float_digits=4),
        "",
    ]
    out_path = _write_markdown_file(
        args.output / "analysis_summary.md", "\n".join(lines)
    )
    logging.getLogger("agripak.data").info("Wrote analysis summary to %s", out_path)
    _print_summary(
        "Statistical Analysis",
        [
            ("Rows", f"{bundle.row_count:,}"),
            ("Target", bundle.schema.target_column),
            ("Leakage flags", str(len(bundle.flagged_leakage_columns))),
            ("Output", str(out_path)),
        ],
    )
    return EXIT_SUCCESS


def _run_features(args: argparse.Namespace) -> int:
    from src.data.analysis import engineer_features
    from src.data.cleaning import clean_dataset
    from src.data.loading import load_dataset, save_dataframe

    _step("Loading dataset...")
    raw_df = load_dataset(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target, on_step=_step)
    _step("Engineering features...")
    fe_result = engineer_features(cleaning_result.data, cleaning_result.schema)
    _step("Saving engineered features...")
    out_path = save_dataframe(fe_result.data, args.output / "engineered_features.csv")
    logging.getLogger("agripak.data").info("Wrote engineered features to %s", out_path)
    _print_summary(
        "Feature Engineering",
        [
            ("Engineered columns", str(len(fe_result.engineered_columns))),
            ("Output", str(out_path)),
        ],
    )
    return EXIT_SUCCESS


def _run_plot(args: argparse.Namespace) -> int:
    _, cleaning_result, bundle = _load_clean_analyze(args)
    plots = _generate_plots_with_eta(cleaning_result.data, bundle, args.assets)
    generated = sum(1 for p in plots if p.generated)
    logging.getLogger("agripak.data").info(
        "Generated %d/%d plots in %s", generated, len(plots), args.assets
    )
    _print_summary(
        "Plot Generation",
        [
            ("Plots generated", f"{generated}/{len(plots)}"),
            ("Assets directory", str(args.assets)),
        ],
    )
    return EXIT_SUCCESS


def _run_report(args: argparse.Namespace) -> int:
    raw_df, cleaning_result, bundle = _load_clean_analyze(args)
    _render_report_bundle(raw_df, cleaning_result, bundle, args)
    report_path = args.output / "README.md"
    logging.getLogger("agripak.data").info("Generated report at %s", report_path)
    _print_summary(
        "Report Generation",
        [
            ("Report", str(report_path)),
            ("Assets", str(args.assets)),
        ],
    )
    return EXIT_SUCCESS


def _run_full(args: argparse.Namespace) -> int:
    from src.data.loading import save_dataframe

    raw_df, cleaning_result, bundle = _load_clean_analyze(args)
    _step("Saving CSV outputs...")
    clean_csv = save_dataframe(cleaning_result.data, args.output / "cleaned_data.csv")
    feat_csv = save_dataframe(
        bundle.feature_engineering.data, args.output / "engineered_features.csv"
    )
    _render_report_bundle(raw_df, cleaning_result, bundle, args)
    report_path = args.output / "README.md"
    logging.getLogger("agripak.data").info(
        "Completed full pipeline into %s", args.output
    )
    _print_summary(
        "Full Pipeline",
        [
            ("Cleaned CSV", str(clean_csv)),
            ("Engineered CSV", str(feat_csv)),
            ("Report", str(report_path)),
            ("Assets", str(args.assets)),
        ],
    )
    return EXIT_SUCCESS


def _dispatch(args: argparse.Namespace) -> int:
    handlers = {
        "run": _run_full,
        "audit": _run_audit,
        "clean": _run_clean,
        "analyze": _run_analyze,
        "features": _run_features,
        "plot": _run_plot,
        "report": _run_report,
    }
    handler = handlers.get(args.command)
    if handler is None:
        raise InvalidArgumentError(f"Unknown command: {args.command}")
    return handler(args)


def _print_help(parser: argparse.ArgumentParser, topic: str | None) -> int:
    if topic is not None:
        try:
            parser.parse_args([topic, "--help"])
        except SystemExit:
            return EXIT_SUCCESS
    _print_summary("AgriPak CLI Commands", _COMMAND_HELP)
    parser.print_help()
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    global _active_step, _pending_summary
    debug = False
    logger = logging.getLogger("agripak.data")
    err_console = Console(stderr=True)
    _pending_summary = None
    try:
        parser = build_parser()
        args = parser.parse_args(list(argv) if argv is not None else None)
        if args.command == "help":
            return _print_help(parser, getattr(args, "topic", None))
        debug = bool(getattr(args, "debug", False))
        from src.data.utils import configure_logging

        logger = configure_logging(
            debug=debug, log_dir=args.log_dir, console=err_console
        )
        logger.info("Running command '%s' on input '%s'", args.command, args.input)
        _validate_args(args)
        start = time.perf_counter()
        total_steps = _COMMAND_STEPS.get(args.command, 1)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(f"Running {args.command}...", total=total_steps)
            started = False

            def _advance(label: str) -> None:
                nonlocal started
                if started:
                    progress.advance(task_id, 1)
                started = True
                progress.update(task_id, description=label)

            _active_step = _advance
            try:
                code = _dispatch(args)
            finally:
                _active_step = None
            progress.update(task_id, completed=total_steps)
        elapsed = time.perf_counter() - start
        logger.info("Completed '%s' in %.2fs", args.command, elapsed)
        if _pending_summary is not None:
            summary_title, summary_rows = _pending_summary
            _pending_summary = None
            _print_summary(summary_title, summary_rows)
        Console().print(f"[bold green]Done[/bold green] in [cyan]{elapsed:.2f}s[/cyan]")
        return code
    except DataPipelineError as exc:
        msg = format_cli_error(exc)
        if debug:
            traceback.print_exc(file=sys.stderr)
        if logger.handlers:
            logger.error("%s", msg)
        else:
            err_console.print(f"[bold red]{msg}[/bold red]")
        return exc.exit_code
    except Exception as exc:
        msg = f"Unexpected error: {exc} (run with --debug for traceback)"
        if debug:
            traceback.print_exc(file=sys.stderr)
        if logger.handlers:
            logger.error("%s", msg)
        else:
            err_console.print(f"[bold red]{msg}[/bold red]")
        return EXIT_UNEXPECTED
