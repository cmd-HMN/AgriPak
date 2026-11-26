from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalysisBundle": ("src.data.analysis", "AnalysisBundle"),
    "AnalysisError": ("src.common.core.errors", "AnalysisError"),
    "CleaningAudit": ("src.data.cleaning", "CleaningAudit"),
    "CleaningEntry": ("src.data.cleaning", "CleaningEntry"),
    "CleaningResult": ("src.data.cleaning", "CleaningResult"),
    "CombinationResult": ("src.data.analysis", "CombinationResult"),
    "DataPipelineError": ("src.common.core.errors", "DataPipelineError"),
    "PlotArtifact": ("src.data.plotting", "PlotArtifact"),
    "QualityAuditResult": ("src.data.analysis", "QualityAuditResult"),
    "analyze_feature_combinations": (
        "src.data.analysis",
        "analyze_feature_combinations",
    ),
    "audit_data_quality": ("src.data.analysis", "audit_data_quality"),
    "build_data_fragment": ("src.data.report", "build_data_fragment"),
    "build_report_markdown": ("src.data.report", "build_report_markdown"),
    "clean_dataset": ("src.data.cleaning", "clean_dataset"),
    "collect_synthetic_notes": ("src.data.analysis", "collect_synthetic_notes"),
    "compute_categorical_summaries": (
        "src.data.analysis",
        "compute_categorical_summaries",
    ),
    "compute_cramers_v": ("src.data.analysis", "compute_cramers_v"),
    "compute_cramers_v_tables": ("src.data.analysis", "compute_cramers_v_tables"),
    "compute_decile_target_curves": (
        "src.data.analysis",
        "compute_decile_target_curves",
    ),
    "compute_numeric_shape_metrics": (
        "src.data.analysis",
        "compute_numeric_shape_metrics",
    ),
    "compute_ranked_target_correlations": (
        "src.data.analysis",
        "compute_ranked_target_correlations",
    ),
    "compute_spearman_matrix": ("src.data.analysis", "compute_spearman_matrix"),
    "compute_subgroup_risk_surfaces": (
        "src.data.analysis",
        "compute_subgroup_risk_surfaces",
    ),
    "compute_target_rates_by_group": (
        "src.data.analysis",
        "compute_target_rates_by_group",
    ),
    "generate_all_plots": ("src.data.plotting", "generate_all_plots"),
    "generate_report": ("src.data.report", "generate_report"),
    "main": ("src.data.cli", "main"),
    "plot_binary_flag_counts": ("src.data.plotting", "plot_binary_flag_counts"),
    "plot_categorical_counts": ("src.data.plotting", "plot_categorical_counts"),
    "plot_continuous_binary_interactions": (
        "src.data.plotting",
        "plot_continuous_binary_interactions",
    ),
    "plot_cramers_v_associations": ("src.data.plotting", "plot_cramers_v_associations"),
    "plot_decile_risk_curves": ("src.data.plotting", "plot_decile_risk_curves"),
    "plot_kruskal_wallis_boxplots": (
        "src.data.plotting",
        "plot_kruskal_wallis_boxplots",
    ),
    "plot_numeric_by_target": ("src.data.plotting", "plot_numeric_by_target"),
    "plot_numeric_distributions": ("src.data.plotting", "plot_numeric_distributions"),
    "plot_pairwise_risk_heatmaps": ("src.data.plotting", "plot_pairwise_risk_heatmaps"),
    "plot_risk_rate_by_category": ("src.data.plotting", "plot_risk_rate_by_category"),
    "plot_spearman_correlation": ("src.data.plotting", "plot_spearman_correlation"),
    "plot_subgroup_risk_surfaces": ("src.data.plotting", "plot_subgroup_risk_surfaces"),
    "plot_target_distribution": ("src.data.plotting", "plot_target_distribution"),
    "plot_top_combinations": ("src.data.plotting", "plot_top_combinations"),
    "run_full_analysis": ("src.data.analysis", "run_full_analysis"),
    "run_kruskal_wallis_grid": ("src.data.analysis", "run_kruskal_wallis_grid"),
    "run_leakage_check": ("src.data.analysis", "run_leakage_check"),
    "run_mann_whitney_tests": ("src.data.analysis", "run_mann_whitney_tests"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'src.data' has no attribute {name!r}")
    mod_name, attr_name = target
    return getattr(import_module(mod_name), attr_name)
