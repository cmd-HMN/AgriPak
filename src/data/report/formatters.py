from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.analysis import AnalysisBundle
from src.data.cleaning import CleaningResult
from src.data.plotting import PlotArtifact
from src.data.schema import DatasetSchema
from src.data.utils import UNIDENTIFIED, dataframe_to_markdown


def _schema_attr(
    schema: DatasetSchema, primary: str, alias: str, default: object
) -> object:
    if hasattr(schema, primary):
        return getattr(schema, primary)
    if hasattr(schema, alias):
        return getattr(schema, alias)
    return default


def _render_plot(
    plot_map: dict[str, PlotArtifact],
    plot_id: str,
    report_dir: Path,
    fallback_title: str,
) -> str:
    plot = plot_map.get(plot_id)
    if plot is None:
        return f"> Plot `{fallback_title}` was not generated."
    if not plot.generated:
        reason = plot.note if plot.note else "no eligible data"
        return f"> **{plot.title}** skipped: {reason}."
    rel_path = os.path.relpath(Path(plot.path), Path(report_dir)).replace(os.sep, "/")
    return f"![{plot.title}]({rel_path})"


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None and bool(pd.notna(value)) else default


def _as_int(value: Any, default: int = 0) -> int:
    return int(value) if value is not None and bool(pd.notna(value)) else default


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or bool(pd.isna(value)):
        return UNIDENTIFIED
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or bool(pd.isna(value)):
        return UNIDENTIFIED
    return f"{float(cast(Any, value)):.{digits}f}"


def _append_table(
    lines: list[str],
    df: pd.DataFrame,
    float_digits: int = 3,
    include_index: bool = False,
    heading: str | None = None,
) -> None:
    if df.empty:
        return
    if heading:
        lines.extend([heading, ""])
    lines.extend(
        [
            dataframe_to_markdown(
                df, float_digits=float_digits, include_index=include_index
            ),
            "",
        ]
    )


def _section_intro(
    heading: str,
    plots_spec: Sequence[tuple[str, str]],
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    lines = [heading, ""]
    for plot_id, fallback in plots_spec:
        lines.extend([_render_plot(plot_map, plot_id, report_dir, fallback), ""])
    return lines


def _build_header(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    analysis: AnalysisBundle,
    plots: Sequence[PlotArtifact],
    input_path: Path,
) -> list[str]:
    schema = analysis.schema
    target = str(_schema_attr(schema, "target_column", "target_col", UNIDENTIFIED))
    generated_plots = sum(1 for plot in plots if plot.generated)
    base_rate = _fmt_pct(analysis.base_rate)
    return [
        '<div align="center">',
        "",
        "# AgriPak Data Analysis Report",
        "",
        "**Dynamically generated exploratory audit, cleaning log, leakage check, and feature-engineering report.**",
        "",
        f"![Rows](https://img.shields.io/badge/rows-{analysis.row_count:,}-blue)",
        f"![Columns](https://img.shields.io/badge/columns-{analysis.col_count}-blue)",
        f"![Plots](https://img.shields.io/badge/plots-{generated_plots}%2F{len(plots)}-brightgreen)",
        f"![Target](https://img.shields.io/badge/target-{target}-orange)",
        f"![Baseline](https://img.shields.io/badge/baseline-{base_rate.replace('%', '%25')}-informational)",
        "",
        "**[1. Overview](#1-dataset-overview--target-balance)** |",
        "**[2. Statistical Audit](#2-data-quality--exploratory-statistical-audit)** |",
        "**[3. Cleaning](#3-data-cleaning-audit)** |",
        "**[4. Leakage Check](#4-leakage-sanity-check)** |",
        "**[5. Feature Engineering](#5-feature-engineering--model-validation)** |",
        "**[6. Exit Codes & Reproducibility](#6-cli-exit-codes--reproducibility)** |",
        "**[Project README](../../README.md)**",
        "",
        "</div>",
        "",
        "---",
        "",
        f"This report is dynamically generated from `{input_path.as_posix()}` and provides two high-level deliverables:",
        "",
        "- Complete exploratory and statistical audit covering distributions, non-parametric tests, correlations, interactions, and independence checks.",
        "- Reproducible cleaning audit, single-feature leakage verification, plot-driven feature engineering, and cross-validated model benchmarks.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Raw input shape | `{len(raw_df):,}` rows x `{len(raw_df.columns)}` cols |",
        f"| Cleaned shape | `{cleaning_result.rows_after:,}` rows x `{analysis.col_count}` cols |",
        f"| Memory footprint | `{cleaning_result.memory_mb_before:.2f} MB` -> `{cleaning_result.memory_mb_after:.2f} MB` |",
        f"| Inferred target | `{target}` (`{base_rate}` baseline) |",
        f"| Generated charts | `{generated_plots}` / `{len(plots)}` |",
        "",
        "---",
        "",
    ]
