from __future__ import annotations

from pathlib import Path

from src.common.dataset.columns import UNIDENTIFIED
from src.data.analysis import AnalysisBundle
from src.data.plotting import PlotArtifact
from src.data.report.formatters import (
    _append_table,
    _as_float,
    _as_int,
    _fmt_float,
    _fmt_pct,
    _schema_attr,
    _section_intro,
)


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
