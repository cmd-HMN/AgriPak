from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.analysis import AnalysisBundle
from src.data.cleaning import CleaningResult
from src.data.errors import EXIT_CODE_TABLE
from src.data.plotting import PlotArtifact
from src.data.report.formatters import (
    _append_table,
    _as_float,
    _as_int,
    _fmt_float,
    _fmt_pct,
    _render_plot,
    _schema_attr,
    _section_intro,
)
from src.data.utils import UNIDENTIFIED, dataframe_to_markdown, is_identified


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


def _build_section_2_1(analysis: AnalysisBundle) -> list[str]:
    qa = analysis.quality_audit
    total_missing = (
        _as_int(qa.column_summary["missing_count"].sum())
        if not qa.column_summary.empty
        else 0
    )
    ws_hits = (
        _as_int(qa.text_hygiene["whitespace_issues"].sum())
        if not qa.text_hygiene.empty
        else 0
    )
    token_hits = (
        _as_int(qa.text_hygiene["missing_token_hits"].sum())
        if not qa.text_hygiene.empty
        else 0
    )
    case_hits = (
        _as_int(qa.text_hygiene["case_variants"].sum())
        if not qa.text_hygiene.empty
        else 0
    )
    lines = ["### 2.1 Nulls, Cardinality, Exact Duplicates & Text Hygiene", ""]
    _append_table(lines, qa.column_summary, float_digits=3)
    _append_table(lines, qa.text_hygiene, float_digits=0)
    lines.extend(
        [
            "**What the output says:**",
            f"- **Completeness:** `{total_missing}` missing values across `{len(qa.column_summary)}` columns and `{qa.duplicate_count}` exact duplicate rows.",
            f"- **Text hygiene:** `{ws_hits}` leading/trailing whitespace issues, `{token_hits}` placeholder tokens, and `{case_hits}` case-variant collisions.",
            "",
        ]
    )
    return lines


def _build_section_2_2_and_2_3(
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    lines = _section_intro(
        "### 2.2 Numeric Distributions & Shape Metrics",
        [("numeric_distributions", "Numeric distributions")],
        plot_map,
        report_dir,
    )
    if not analysis.numeric_shape.empty:
        _append_table(lines, analysis.numeric_shape, float_digits=3)
        lines.append("**What the output says:**")
        for _, row in analysis.numeric_shape.iterrows():
            lines.append(
                f"- `{row['column']}` spans `{_fmt_float(row.get('min'), 2)}` to `{_fmt_float(row.get('max'), 2)}` "
                f"(median `{_fmt_float(row.get('median'), 2)}`, skew `{_fmt_float(row.get('skewness'))}`, "
                f"excess kurtosis `{_fmt_float(row.get('kurtosis'))}`)."
            )
    else:
        lines.append("No numeric columns available for shape diagnostics.")
    lines.append("")
    lines.extend(
        _section_intro(
            "### 2.3 Categorical & Binary Feature Distributions",
            [
                ("categorical_counts", "Categorical counts"),
                ("binary_flag_counts", "Binary flag counts"),
            ],
            plot_map,
            report_dir,
        )
    )
    lines.append("**What the output says:**")
    if analysis.categorical_counts:
        for col, tbl in analysis.categorical_counts.items():
            if tbl.empty:
                continue
            top_row = tbl.iloc[0]
            lines.append(
                f"- `{col}` has `{len(tbl)}` levels; level shares range from `{_as_float(tbl['pct'].min()):.1f}%` to `{_as_float(tbl['pct'].max()):.1f}%` "
                f"(largest: `{top_row.get('category', top_row.iloc[0])}` at `{_as_float(top_row['pct']):.1f}%`)."
            )
    else:
        lines.append("No categorical or binary columns detected.")
    lines.append("")
    return lines


def _build_section_2_4_and_2_5(
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    lines = _section_intro(
        "### 2.4 Target Rate by Categorical & Binary Features",
        [("risk_rate_by_category", "Risk rate by category")],
        plot_map,
        report_dir,
    )
    lines.append("**What the output says:**")
    if analysis.target_rates_by_group:
        for col, tbl in analysis.target_rates_by_group.items():
            if tbl.empty:
                continue
            rate_col = "target_rate" if "target_rate" in tbl.columns else "rate"
            group_col = "group" if "group" in tbl.columns else str(tbl.columns[0])
            max_row = tbl.sort_values(rate_col, ascending=False).iloc[0]
            min_row = tbl.sort_values(rate_col, ascending=True).iloc[0]
            hi, lo = _as_float(max_row[rate_col]), _as_float(min_row[rate_col])
            lines.append(
                f"- `{col}` target rate spans `{_fmt_pct(lo)}` (`{min_row[group_col]}`) "
                f"to `{_fmt_pct(hi)}` (`{max_row[group_col]}`), spread `{(hi - lo) * 100:.1f}` pp."
            )
    else:
        lines.append(f"Target rates by group: `{UNIDENTIFIED}`.")
    lines.append("")
    lines.extend(
        _section_intro(
            "### 2.5 Numeric Distributions by Target Class & Mann-Whitney U Tests",
            [("numeric_by_target", "Numeric by target")],
            plot_map,
            report_dir,
        )
    )
    if not analysis.mann_whitney.empty:
        _append_table(lines, analysis.mann_whitney, float_digits=4)
        lines.append("**What the output says:**")
        for _, row in analysis.mann_whitney.iterrows():
            med0 = row.get("median_neg", row.get("median_0"))
            med1 = row.get("median_pos", row.get("median_1"))
            q_val = _as_float(row.get("p_adj_fdr", row.get("q_fdr", 1.0)), 1.0)
            lines.append(
                f"- `{row['feature']}`: median `{_fmt_float(med0, 2)}` (class 0) vs `{_fmt_float(med1, 2)}` (class 1), "
                f"rank-biserial `{_fmt_float(row.get('rank_biserial'), 3)}`, FDR q-value `{q_val:.3e}`."
            )
    else:
        lines.append(f"Mann-Whitney U results: `{UNIDENTIFIED}`.")
    lines.append("")
    return lines


def _build_section_2_6_and_2_7(
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    lines = _section_intro(
        "### 2.6 Spearman Correlation & Decile Target Curves",
        [
            ("spearman_correlation", "Spearman correlation"),
            ("decile_risk_curves", "Decile risk curves"),
        ],
        plot_map,
        report_dir,
    )
    if not analysis.ranked_target_spearman.empty:
        _append_table(lines, analysis.ranked_target_spearman, float_digits=4)
        lines.append("**What the output says:**")
        top_corr = analysis.ranked_target_spearman.iloc[0]
        rho_col = (
            "spearman_r"
            if "spearman_r" in analysis.ranked_target_spearman.columns
            else "spearman_rho"
        )
        lines.append(
            f"- Strongest monotonic predictor of target is `{top_corr['feature']}` (Spearman `rho = {_fmt_float(top_corr[rho_col], 3)}`)."
        )
        for col, curve in analysis.decile_curves.items():
            if curve.empty:
                continue
            rate_col = "target_rate" if "target_rate" in curve.columns else "rate"
            lines.append(
                f"- `{col}` decile target rate ranges from `{_as_float(curve[rate_col].min()) * 100:.1f}%` to `{_as_float(curve[rate_col].max()) * 100:.1f}%`."
            )
    else:
        lines.append(f"Spearman target ranking: `{UNIDENTIFIED}`.")
    lines.append("")
    combos = analysis.combinations
    lines.extend(
        _section_intro(
            "### 2.7 Pairwise Interactions & Condition Combinations",
            [
                ("pairwise_risk_heatmaps", "Pairwise risk heatmaps"),
                ("top_combinations", "Top combinations"),
            ],
            plot_map,
            report_dir,
        )
    )
    lines.extend(
        [
            f"Evaluated `{combos.total_combos:,}` multi-feature combinations with `n >= {analysis.min_n}`.",
            "",
        ]
    )
    _append_table(
        lines,
        combos.top_risky.head(10),
        float_digits=3,
        heading="#### Top Elevated-Risk Combinations",
    )
    _append_table(
        lines,
        combos.top_protective.head(10),
        float_digits=3,
        heading="#### Top Protective Combinations",
    )
    _append_table(
        lines,
        combos.top_synergy.head(10),
        float_digits=3,
        heading="#### Strongest Non-Additive Synergies",
    )
    lines.extend(
        [
            "**What the output says:**",
            f"- Train/test split replication rate across top combinations: `{combos.replication_rate * 100:.1f}%`.",
            "",
        ]
    )
    return lines


def _build_section_2_8_and_2_9(
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    schema = analysis.schema
    lines = _section_intro(
        "### 2.8 Continuous x Binary Interactions & Subgroup Stability",
        [
            ("continuous_binary_interactions", "Continuous binary interactions"),
            ("subgroup_risk_surfaces", "Subgroup risk surfaces"),
        ],
        plot_map,
        report_dir,
    )
    lines.extend(
        [
            "**What the output says:**",
            f"- Primary continuous driver `{getattr(schema, 'primary_driver', UNIDENTIFIED)}` paired with secondary driver `{getattr(schema, 'secondary_driver', UNIDENTIFIED)}` across subgroup `{_schema_attr(schema, 'primary_subgroup', 'subgroup_col', UNIDENTIFIED)}`.",
        ]
    )
    if analysis.subgroup_surfaces:
        lines.append(
            f"- Computed subgroup risk surfaces across `{len(analysis.subgroup_surfaces)}` subgroup levels (each cell filtered by `n >= {analysis.min_n}`)."
        )
    else:
        lines.append(f"- Subgroup risk surfaces: `{UNIDENTIFIED}`.")
    lines.append("")
    lines.extend(
        _section_intro(
            "### 2.9 Kruskal-Wallis Group Tests & Cramer's V Associations",
            [
                ("kruskal_wallis_boxplots", "Kruskal-Wallis boxplots"),
                ("cramers_v_associations", "Cramer's V associations"),
            ],
            plot_map,
            report_dir,
        )
    )
    _append_table(lines, analysis.kruskal_wallis, float_digits=4)
    _append_table(lines, analysis.target_cramers_v, float_digits=4)
    lines.append("**What the output says:**")
    if not analysis.kruskal_wallis.empty:
        fdr_col = (
            "p_adj_fdr" if "p_adj_fdr" in analysis.kruskal_wallis.columns else "q_fdr"
        )
        sig_kw = (
            _as_int((analysis.kruskal_wallis[fdr_col] < 0.05).sum())
            if fdr_col in analysis.kruskal_wallis.columns
            else 0
        )
        lines.append(
            f"- `{sig_kw}` of `{len(analysis.kruskal_wallis)}` numeric-by-categorical Kruskal-Wallis pairs are significant after FDR correction (`q < 0.05`)."
        )
    if not analysis.target_cramers_v.empty:
        top_v = analysis.target_cramers_v.iloc[0]
        lines.append(
            f"- Highest categorical/binary Cramer's V with target: `{top_v['feature']}` (`V = {_fmt_float(top_v['cramers_v'], 3)}`)."
        )
    lines.append("")
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


def _build_section_5(
    analysis: AnalysisBundle,
    plot_map: dict[str, PlotArtifact],
    report_dir: Path,
) -> list[str]:
    fe = analysis.feature_engineering
    lines = [
        "## 5. Feature Engineering & Model Validation",
        "",
        f"- **Engineered columns:** `{', '.join(fe.engineered_columns) if fe.engineered_columns else UNIDENTIFIED}`",
        f"- **Raw signal columns:** `{', '.join(fe.raw_signal_columns) if fe.raw_signal_columns else UNIDENTIFIED}`",
        f"- **Inferred risk threshold:** `{_fmt_float(fe.inferred_threshold, 2)}` (`{fe.threshold_direction}`)",
        "",
        "### 5.1 Engineered Feature Diagnostics & Target Correlations",
        "",
    ]
    _append_table(lines, fe.summary_stats, float_digits=3, include_index=True)
    _append_table(lines, fe.correlation_comparison, float_digits=4)
    lines.extend(
        [
            "### 5.2 Visual Check: Engineered Risk Curves",
            "",
            _render_plot(
                plot_map, "engineered_features", report_dir, "Engineered features"
            ),
            "",
            "### 5.3 Cross-Validated ROC-AUC Comparison",
            "",
        ]
    )
    if not analysis.cv_auc_table.empty:
        _append_table(lines, analysis.cv_auc_table, float_digits=4)
        lines.append("**How to read the CV AUC table:**")
        for _, row in analysis.cv_auc_table.iterrows():
            lines.append(
                f"- `{row['model_set']}` (`{_as_int(row['n_features'])}` features): mean ROC-AUC `{_as_float(row['mean_auc']):.4f}` "
                f"(std `{_as_float(row['std_auc']):.4f}`, range `{_as_float(row['min_auc']):.4f}`–`{_as_float(row['max_auc']):.4f}`)."
            )
    else:
        lines.append(f"CV AUC validation: `{UNIDENTIFIED}`.")
    lines.append("")
    return lines


def _build_section_6() -> list[str]:
    exit_df = pd.DataFrame(
        [
            {"code": code, "error_class": name, "meaning": desc}
            for code, name, desc in EXIT_CODE_TABLE
        ]
    )
    return [
        "## 6. CLI Exit Codes & Reproducibility",
        "",
        dataframe_to_markdown(exit_df, float_digits=0),
        "",
        "Reproduce this report via:",
        "",
        "```bash",
        "python -m src.data report --input data/raw --output src/data --assets src/assets",
        "```",
        "",
    ]
