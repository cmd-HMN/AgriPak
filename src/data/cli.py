from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pandas as pd

from src.data.analysis import (
    AnalysisBundle,
    audit_data_quality,
    engineer_features,
    run_full_analysis,
)
from src.data.cleaning import CleaningResult, clean_dataset
from src.data.errors import (
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    DataPipelineError,
    InvalidArgumentError,
    ReportWriteError,
    format_cli_error,
)
from src.data.loading import load_dataset, load_raw_text, save_dataframe
from src.data.plotting import generate_all_plots
from src.data.report import generate_report
from src.data.schema import infer_schema
from src.data.utils import configure_logging, dataframe_to_markdown, ensure_directory


class _CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise InvalidArgumentError(message)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("src/data"))
    parser.add_argument("--assets", type=Path, default=Path("src/assets"))
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--leak-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = _CLIArgumentParser(prog="src.data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "audit", "clean", "analyze", "features", "plot", "report"):
        _add_common_arguments(subparsers.add_parser(name))
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


def _write_markdown_file(path: Path, content: str) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(f"Failed to write '{target}': {exc}") from exc
    return target


def _load_clean_analyze(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, CleaningResult, AnalysisBundle]:
    raw_df = load_dataset(args.input)
    raw_text_df = load_raw_text(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target)
    bundle = run_full_analysis(
        cleaning_result.data,
        cleaning_result.schema,
        raw_text_df=raw_text_df,
        min_n=args.min_n,
        leak_threshold=args.leak_threshold,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    return raw_df, cleaning_result, bundle


def _render_report_bundle(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    bundle: AnalysisBundle,
    args: argparse.Namespace,
) -> None:
    plots = generate_all_plots(cleaning_result.data, bundle, args.assets)
    generate_report(
        raw_df=raw_df,
        cleaning_result=cleaning_result,
        analysis=bundle,
        plots=plots,
        input_path=args.input,
        report_path=args.output / "README.md",
    )


def _run_audit(args: argparse.Namespace) -> int:
    raw_df = load_dataset(args.input)
    raw_text_df = load_raw_text(args.input)
    schema = infer_schema(raw_df, target_column=args.target)
    qa = audit_data_quality(raw_df, raw_text_df=raw_text_df)
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
    _write_markdown_file(args.output / "quality_audit.md", "\n".join(lines))
    return EXIT_SUCCESS


def _run_clean(args: argparse.Namespace) -> int:
    raw_df = load_dataset(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target)
    save_dataframe(cleaning_result.data, args.output / "cleaned_data.csv")
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
    _write_markdown_file(args.output / "cleaning_audit.md", "\n".join(lines))
    return EXIT_SUCCESS


def _run_analyze(args: argparse.Namespace) -> int:
    _, _, bundle = _load_clean_analyze(args)
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
    _write_markdown_file(args.output / "analysis_summary.md", "\n".join(lines))
    return EXIT_SUCCESS


def _run_features(args: argparse.Namespace) -> int:
    raw_df = load_dataset(args.input)
    cleaning_result = clean_dataset(raw_df, target_column=args.target)
    fe_result = engineer_features(cleaning_result.data, cleaning_result.schema)
    save_dataframe(fe_result.data, args.output / "engineered_features.csv")
    return EXIT_SUCCESS


def _run_plot(args: argparse.Namespace) -> int:
    _, cleaning_result, bundle = _load_clean_analyze(args)
    generate_all_plots(cleaning_result.data, bundle, args.assets)
    return EXIT_SUCCESS


def _run_report(args: argparse.Namespace) -> int:
    raw_df, cleaning_result, bundle = _load_clean_analyze(args)
    _render_report_bundle(raw_df, cleaning_result, bundle, args)
    return EXIT_SUCCESS


def _run_full(args: argparse.Namespace) -> int:
    raw_df, cleaning_result, bundle = _load_clean_analyze(args)
    save_dataframe(cleaning_result.data, args.output / "cleaned_data.csv")
    save_dataframe(
        bundle.feature_engineering.data, args.output / "engineered_features.csv"
    )
    _render_report_bundle(raw_df, cleaning_result, bundle, args)
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


def main(argv: Sequence[str] | None = None) -> int:
    debug = False
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
        debug = bool(getattr(args, "debug", False))
        configure_logging(debug=debug)
        _validate_args(args)
        return _dispatch(args)
    except DataPipelineError as exc:
        if debug:
            traceback.print_exc(file=sys.stderr)
        print(format_cli_error(exc), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        if debug:
            traceback.print_exc(file=sys.stderr)
        print(
            f"Unexpected error: {exc} (run with --debug for traceback)", file=sys.stderr
        )
        return EXIT_UNEXPECTED
