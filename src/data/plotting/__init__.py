from __future__ import annotations

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
from src.data.plotting.generate import generate_all_plots
from src.common.report.plotting import PlotArtifact

__all__ = [
    "PlotArtifact",
    "generate_all_plots",
    "plot_binary_flag_counts",
    "plot_categorical_counts",
    "plot_continuous_binary_interactions",
    "plot_cramers_v_associations",
    "plot_decile_risk_curves",
    "plot_kruskal_wallis_boxplots",
    "plot_numeric_by_target",
    "plot_numeric_distributions",
    "plot_pairwise_risk_heatmaps",
    "plot_risk_rate_by_category",
    "plot_spearman_correlation",
    "plot_subgroup_risk_surfaces",
    "plot_target_distribution",
    "plot_top_combinations",
]
