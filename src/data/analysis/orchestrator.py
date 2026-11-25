from __future__ import annotations

from typing import cast

import pandas as pd

from src.data.analysis.interactions import (
    analyze_feature_combinations,
    compute_subgroup_risk_surfaces,
)
from src.data.analysis.modeling import (
    _collect_synthetic_notes,
    engineer_features,
    run_leakage_check,
    validate_models_cv,
)
from src.data.analysis.stats import (
    audit_data_quality,
    compute_categorical_summaries,
    compute_cramers_v_tables,
    compute_decile_target_curves,
    compute_numeric_shape_metrics,
    compute_ranked_target_correlations,
    compute_spearman_matrix,
    compute_target_rates_by_group,
    run_kruskal_wallis_grid,
    run_mann_whitney_tests,
)
from src.data.analysis.types import (
    AnalysisBundle,
    _flt,
    _get_drivers,
    _num,
    _target_series,
)
from src.data.errors import AnalysisError
from src.data.schema import DatasetSchema, resolve_drivers
from src.data.utils import UNIDENTIFIED, compute_min_subgroup_size, is_identified


def run_full_analysis(
    clean_df: pd.DataFrame,
    schema: DatasetSchema,
    raw_text_df: pd.DataFrame | None = None,
    min_n: int | None = None,
    leak_threshold: float = 0.90,
    n_splits: int = 5,
    random_state: int = 42,
) -> AnalysisBundle:
    if not isinstance(clean_df, pd.DataFrame):
        raise AnalysisError("clean_df must be a valid pandas DataFrame.")
    r_cnt, c_cnt = len(clean_df), len(clean_df.columns)
    mn = min_n if min_n is not None else compute_min_subgroup_size(r_cnt)
    rs = resolve_drivers(clean_df, schema)
    tc, ts = rs.target_column, _target_series(clean_df, rs.target_column)
    qa = audit_data_quality(clean_df, raw_text_df=raw_text_df)
    nc = [c for c in rs.numeric_columns if c in clean_df.columns]
    cc = [c for c in rs.categorical if c in clean_df.columns]
    bc = [c for c in rs.binary_columns if c in clean_df.columns]
    cbc = cc + bc
    shape = compute_numeric_shape_metrics(clean_df, nc)
    sp_mat = compute_spearman_matrix(
        clean_df, list(dict.fromkeys(nc + bc + ([tc] if is_identified(tc) else [])))
    )
    combos = analyze_feature_combinations(clean_df, rs, mn, random_state=random_state)
    dx, dy = _get_drivers(rs, df=clean_df)
    sub_col = (
        rs.primary_subgroup
        if is_identified(rs.primary_subgroup)
        else (cc[0] if cc else UNIDENTIFIED)
    )
    tv, pv = compute_cramers_v_tables(clean_df, cbc, tc)
    leak_df, flagged = run_leakage_check(
        clean_df,
        rs,
        leak_threshold=leak_threshold,
        n_splits=n_splits,
        random_state=random_state,
    )
    fe = engineer_features(clean_df, rs)
    num_desc = (
        cast(
            pd.DataFrame, pd.DataFrame({c: _num(clean_df[c]) for c in nc}).describe().T
        )
        if nc
        else pd.DataFrame()
    )
    return AnalysisBundle(
        rs,
        r_cnt,
        c_cnt,
        _flt(ts.mean()) if ts is not None else None,
        mn,
        qa,
        num_desc,
        shape,
        compute_categorical_summaries(clean_df, cbc),
        compute_target_rates_by_group(clean_df, cbc, tc),
        run_mann_whitney_tests(clean_df, nc + bc, tc),
        sp_mat,
        compute_ranked_target_correlations(sp_mat, tc),
        compute_decile_target_curves(clean_df, nc, tc),
        combos,
        compute_subgroup_risk_surfaces(clean_df, dx, dy, tc, sub_col, mn),
        run_kruskal_wallis_grid(clean_df, nc, cc),
        tv,
        pv,
        leak_df,
        flagged,
        _collect_synthetic_notes(qa, shape, flagged, combos),
        fe,
        validate_models_cv(
            clean_df, rs, fe, n_splits=n_splits, random_state=random_state
        ),
    )
