from __future__ import annotations

from src.data.analysis.associations import (
    compute_cramers_v,
    compute_cramers_v_tables,
    compute_ranked_target_correlations,
    compute_spearman_matrix,
    run_kruskal_wallis_grid,
    run_mann_whitney_tests,
)
from src.data.analysis.interactions import (
    analyze_feature_combinations,
    compute_subgroup_risk_surfaces,
)
from src.data.analysis.leakage import (
    collect_synthetic_notes,
    run_leakage_check,
)
from src.data.analysis.orchestrator import run_full_analysis
from src.data.analysis.profiling import (
    audit_data_quality,
    compute_categorical_summaries,
    compute_decile_target_curves,
    compute_numeric_shape_metrics,
    compute_target_rates_by_group,
)
from src.data.analysis.types import (
    AnalysisBundle,
    CombinationResult,
    QualityAuditResult,
)

__all__ = [
    "AnalysisBundle",
    "CombinationResult",
    "QualityAuditResult",
    "analyze_feature_combinations",
    "audit_data_quality",
    "collect_synthetic_notes",
    "compute_categorical_summaries",
    "compute_cramers_v",
    "compute_cramers_v_tables",
    "compute_decile_target_curves",
    "compute_numeric_shape_metrics",
    "compute_ranked_target_correlations",
    "compute_spearman_matrix",
    "compute_subgroup_risk_surfaces",
    "compute_target_rates_by_group",
    "run_full_analysis",
    "run_kruskal_wallis_grid",
    "run_leakage_check",
    "run_mann_whitney_tests",
]
