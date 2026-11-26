from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pandas as pd

from src.data.analysis.interactions import (
    analyze_feature_combinations,
    compute_subgroup_risk_surfaces,
)
from src.data.analysis.leakage import collect_synthetic_notes, run_leakage_check
from src.data.analysis.associations import (
    compute_cramers_v_tables,
    compute_ranked_target_correlations,
    compute_spearman_matrix,
    run_kruskal_wallis_grid,
    run_mann_whitney_tests,
)
from src.data.analysis.profiling import (
    audit_data_quality,
    compute_categorical_summaries,
    compute_decile_target_curves,
    compute_numeric_shape_metrics,
    compute_target_rates_by_group,
)
from src.data.analysis.types import (
    AnalysisBundle,
    _flt,
    _get_drivers,
    _num,
    _target_series,
)
from src.common.dataset.columns import (
    UNIDENTIFIED,
    DatasetSchema,
    is_identified,
    resolve_drivers,
)
from src.common.core.validation import compute_min_subgroup_size
from src.common.core.errors import AnalysisError


def run_full_analysis(
    clean_df: pd.DataFrame,
    schema: DatasetSchema,
    raw_text_df: pd.DataFrame | None = None,
    min_n: int | None = None,
    leak_threshold: float = 0.90,
    n_splits: int = 5,
    random_state: int = 42,
    on_step: Callable[[str], None] | None = None,
) -> AnalysisBundle:
    if not isinstance(clean_df, pd.DataFrame):
        raise AnalysisError("clean_df must be a valid pandas DataFrame.")

    def _tick(label: str) -> None:
        if on_step is not None:
            on_step(label)

    r_cnt, c_cnt = len(clean_df), len(clean_df.columns)
    mn = min_n if min_n is not None else compute_min_subgroup_size(r_cnt)
    rs = resolve_drivers(clean_df, schema)
    tc, ts = rs.target_column, _target_series(clean_df, rs.target_column)
    _tick("Auditing data quality...")
    qa = audit_data_quality(clean_df, raw_text_df=raw_text_df)
    nc = [c for c in rs.numeric_columns if c in clean_df.columns]
    cc = [c for c in rs.categorical if c in clean_df.columns]
    bc = [c for c in rs.binary_columns if c in clean_df.columns]
    cbc = cc + bc
    _tick("Computing numeric shape metrics...")
    shape = compute_numeric_shape_metrics(clean_df, nc)
    _tick("Computing Spearman correlations...")
    sp_mat = compute_spearman_matrix(
        clean_df, list(dict.fromkeys(nc + bc + ([tc] if is_identified(tc) else [])))
    )
    _tick("Analyzing feature combinations...")
    combos = analyze_feature_combinations(clean_df, rs, mn, random_state=random_state)
    dx, dy = _get_drivers(rs, df=clean_df)
    sub_col = (
        rs.primary_subgroup
        if is_identified(rs.primary_subgroup)
        else (cc[0] if cc else UNIDENTIFIED)
    )
    _tick("Computing Cramer's V associations...")
    tv, pv = compute_cramers_v_tables(clean_df, cbc, tc)
    _tick("Running leakage check...")
    leak_df, flagged = run_leakage_check(clean_df, rs, threshold=leak_threshold)
    num_desc = (
        cast(
            pd.DataFrame, pd.DataFrame({c: _num(clean_df[c]) for c in nc}).describe().T
        )
        if nc
        else pd.DataFrame()
    )
    _tick("Computing group target rates...")
    cat_counts = compute_categorical_summaries(clean_df, cbc)
    group_rates = compute_target_rates_by_group(clean_df, cbc, tc)
    _tick("Running Mann-Whitney U tests...")
    mw = run_mann_whitney_tests(clean_df, nc + bc, tc)
    ranked_sp = compute_ranked_target_correlations(sp_mat, tc)
    deciles = compute_decile_target_curves(clean_df, nc, tc)
    _tick("Computing subgroup risk surfaces...")
    surfaces = compute_subgroup_risk_surfaces(clean_df, dx, dy, tc, sub_col, mn)
    _tick("Running Kruskal-Wallis tests...")
    kw = run_kruskal_wallis_grid(clean_df, nc, cc)
    return AnalysisBundle(
        rs,
        r_cnt,
        c_cnt,
        _flt(ts.mean()) if ts is not None else None,
        mn,
        qa,
        num_desc,
        shape,
        cat_counts,
        group_rates,
        mw,
        sp_mat,
        ranked_sp,
        deciles,
        combos,
        surfaces,
        kw,
        tv,
        pv,
        leak_df,
        flagged,
        collect_synthetic_notes(qa, shape, flagged, combos),
    )
