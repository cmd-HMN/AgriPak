from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.dataset.columns import UNIDENTIFIED, is_identified
from src.common.report.markdown import dataframe_to_markdown
from src.data.analysis import AnalysisBundle
from src.data.cleaning import CleaningResult
from src.data.plotting import PlotArtifact
from src.data.report.formatters import (
    _append_table,
    _fmt_pct,
    _render_plot,
    _schema_attr,
)


def _build_section_1(
    raw_df: pd.DataFrame,
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
    input_path: Path,
) -> list[str]:
    schema = analysis.schema
    target = str(_schema_attr(schema, "target_column", "target_col", UNIDENTIFIED))
    target_kind = str(getattr(schema, "target_kind", UNIDENTIFIED))
    primary = str(getattr(schema, "primary_driver", UNIDENTIFIED))
    secondary = str(getattr(schema, "secondary_driver", UNIDENTIFIED))
    subgroup = str(
        _schema_attr(schema, "primary_subgroup", "subgroup_col", UNIDENTIFIED)
    )
    lines = [
        "## 1. Dataset Overview & Target Balance",
        "",
        f"Source input: `{input_path.as_posix()}`",
        "",
        f"- **Dimensions:** `{analysis.row_count:,}` rows x `{analysis.col_count}` columns (raw input `{len(raw_df):,}` x `{len(raw_df.columns)}`)",
        f"- **Minimum subgroup threshold (`min_n`):** `{analysis.min_n}`",
        f"- **Baseline positive rate:** `{_fmt_pct(analysis.base_rate)}`",
        f"- **Target column:** `{target}` (`{target_kind}`)",
        f"- **Primary driver:** `{primary}`",
        f"- **Secondary driver:** `{secondary}`",
        f"- **Primary subgroup:** `{subgroup}`",
        "",
        "### 1.1 Schema & Column Profiles",
        "",
    ]
    if schema.profiles:
        profile_df = pd.DataFrame(
            [
                {
                    "column": p.name,
                    "kind": p.kind,
                    "role": p.role,
                    "dtype": p.dtype,
                    "nunique": p.nunique,
                    "null_count": p.null_count,
                    "null_pct": p.null_pct,
                }
                for p in schema.profiles
            ]
        )
        _append_table(lines, profile_df, float_digits=2)
    else:
        lines.extend([f"Column profiles: {UNIDENTIFIED}.", ""])
    lines.extend(["### 1.2 Numeric Summary & Target Distribution", ""])
    if not analysis.numeric_describe.empty:
        _append_table(
            lines, analysis.numeric_describe, float_digits=3, include_index=True
        )
    else:
        lines.extend(["No numeric columns detected for summary statistics.", ""])
    lines.extend(
        [
            _render_plot(
                plot_map, "target_distribution", report_dir, "Target distribution"
            ),
            "",
            "**What the output says:**",
        ]
    )
    if is_identified(target) and analysis.base_rate is not None:
        pos_pct = analysis.base_rate * 100.0
        lines.append(
            f"- `{target}` has `{pos_pct:.1f}%` positive class and `{100.0 - pos_pct:.1f}%` negative class across `{analysis.row_count:,}` rows."
        )
    else:
        lines.append(
            f"- Target distribution is `{UNIDENTIFIED}` (no binary target column identified)."
        )
    lines.extend(
        [
            f"- Raw dataset memory footprint before optimization: `{analysis.quality_audit.memory_mb:.2f} MB`.",
            "",
        ]
    )
    return lines


def _build_section_3(cleaning_result: CleaningResult) -> list[str]:
    mem_before, mem_after = (
        cleaning_result.memory_mb_before,
        cleaning_result.memory_mb_after,
    )
    reduction_pct = (
        ((mem_before - mem_after) / mem_before * 100.0) if mem_before > 0 else 0.0
    )
    return [
        "## 3. Data Cleaning Audit",
        "",
        dataframe_to_markdown(cleaning_result.audit.to_dataframe(), float_digits=0),
        "",
        "**What the output says:**",
        f"- **Row preservation:** `{cleaning_result.rows_before:,}` rows in -> `{cleaning_result.rows_after:,}` rows out.",
        f"- **Memory optimization:** `{mem_before:.2f} MB` -> `{mem_after:.2f} MB` (`{reduction_pct:.1f}%` reduction via compact dtypes).",
        "",
    ]


def _build_section_4(analysis: AnalysisBundle) -> list[str]:
    lines = ["## 4. Leakage Sanity Check", ""]
    _append_table(lines, analysis.leakage_table, float_digits=4)
    lines.append("**What the output says:**")
    if analysis.flagged_leakage_columns:
        lines.append(
            f"- Flagged potential leakage columns: `{', '.join(analysis.flagged_leakage_columns)}`."
        )
    else:
        lines.append("- Zero columns exceeded the leakage threshold.")
    lines.extend(f"- {note}" for note in analysis.synthetic_notes)
    lines.append("")
    return lines
