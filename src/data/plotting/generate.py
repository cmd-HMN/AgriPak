from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    plot_engineered_features,
    plot_kruskal_wallis_boxplots,
    plot_pairwise_risk_heatmaps,
    plot_subgroup_risk_surfaces,
    plot_top_combinations,
)
from src.data.plotting.helpers import PlotArtifact


def generate_all_plots(
    clean_df: pd.DataFrame,
    analysis: AnalysisBundle,
    assets_dir: Path,
) -> list[PlotArtifact]:
    schema = analysis.schema
    base_rate = analysis.base_rate
    return [
        plot_target_distribution(clean_df, schema, assets_dir),
        plot_numeric_distributions(clean_df, schema, assets_dir),
        plot_categorical_counts(clean_df, schema, assets_dir),
        plot_binary_flag_counts(clean_df, schema, assets_dir),
        plot_risk_rate_by_category(clean_df, schema, base_rate, assets_dir),
        plot_numeric_by_target(clean_df, schema, assets_dir),
        plot_spearman_correlation(analysis.spearman_matrix, assets_dir),
        plot_decile_risk_curves(
            analysis.decile_curves, base_rate, schema.target_column, assets_dir
        ),
        plot_pairwise_risk_heatmaps(
            clean_df, schema, base_rate, analysis.min_n, assets_dir
        ),
        plot_top_combinations(
            analysis.combinations, base_rate, schema.target_column, assets_dir
        ),
        plot_continuous_binary_interactions(clean_df, schema, base_rate, assets_dir),
        plot_subgroup_risk_surfaces(
            analysis.subgroup_surfaces, schema, base_rate, assets_dir
        ),
        plot_kruskal_wallis_boxplots(
            clean_df, schema, analysis.kruskal_wallis, assets_dir
        ),
        plot_cramers_v_associations(
            analysis.target_cramers_v,
            analysis.pairwise_cramers_v,
            schema.target_column,
            assets_dir,
        ),
        plot_engineered_features(
            analysis.feature_engineering, schema, base_rate, assets_dir
        ),
    ]
