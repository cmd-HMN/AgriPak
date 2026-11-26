from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.common.report.plotting import PlotArtifact, PlotSpec, run_plot_jobs
from src.data.analysis import AnalysisBundle
from src.data.plotting.charts_eda import (
    plot_binary_flag_counts,
    plot_categorical_counts,
    plot_decile_risk_curves,
    plot_numeric_by_target,
    plot_numeric_distributions,
    plot_risk_rate_by_category,
    plot_spearman_correlation,
    plot_target_distribution,
)
from src.data.plotting.charts_interactions import (
    plot_continuous_binary_interactions,
    plot_cramers_v_associations,
    plot_kruskal_wallis_boxplots,
    plot_pairwise_risk_heatmaps,
    plot_subgroup_risk_surfaces,
    plot_top_combinations,
)


def generate_all_plots(
    clean_df: pd.DataFrame,
    analysis: AnalysisBundle,
    assets_dir: Path,
    on_plot: Callable[[str], None] | None = None,
) -> list[PlotArtifact]:
    schema = analysis.schema
    base_rate = analysis.base_rate
    raw_jobs: list[tuple[str, Callable[[], PlotArtifact]]] = [
        (
            "target_distribution",
            lambda: plot_target_distribution(clean_df, schema, assets_dir),
        ),
        (
            "numeric_distributions",
            lambda: plot_numeric_distributions(clean_df, schema, assets_dir),
        ),
        (
            "categorical_counts",
            lambda: plot_categorical_counts(clean_df, schema, assets_dir),
        ),
        (
            "binary_flag_counts",
            lambda: plot_binary_flag_counts(clean_df, schema, assets_dir),
        ),
        (
            "risk_rate_by_category",
            lambda: plot_risk_rate_by_category(clean_df, schema, base_rate, assets_dir),
        ),
        (
            "numeric_by_target",
            lambda: plot_numeric_by_target(clean_df, schema, assets_dir),
        ),
        (
            "spearman_correlation",
            lambda: plot_spearman_correlation(analysis.spearman_matrix, assets_dir),
        ),
        (
            "decile_risk_curves",
            lambda: plot_decile_risk_curves(
                analysis.decile_curves, base_rate, schema.target_column, assets_dir
            ),
        ),
        (
            "pairwise_risk_heatmaps",
            lambda: plot_pairwise_risk_heatmaps(
                clean_df, schema, base_rate, analysis.min_n, assets_dir
            ),
        ),
        (
            "top_combinations",
            lambda: plot_top_combinations(
                analysis.combinations, base_rate, schema.target_column, assets_dir
            ),
        ),
        (
            "continuous_binary_interactions",
            lambda: plot_continuous_binary_interactions(
                clean_df, schema, base_rate, assets_dir
            ),
        ),
        (
            "subgroup_risk_surfaces",
            lambda: plot_subgroup_risk_surfaces(
                analysis.subgroup_surfaces, schema, base_rate, assets_dir
            ),
        ),
        (
            "kruskal_wallis_boxplots",
            lambda: plot_kruskal_wallis_boxplots(
                clean_df, schema, analysis.kruskal_wallis, assets_dir
            ),
        ),
        (
            "cramers_v_associations",
            lambda: plot_cramers_v_associations(
                analysis.target_cramers_v,
                analysis.pairwise_cramers_v,
                schema.target_column,
                assets_dir,
            ),
        ),
    ]
    jobs = [(PlotSpec(n, f"{n}.png", n, "EDA"), fn) for n, fn in raw_jobs]
    return run_plot_jobs(jobs, on_plot)
