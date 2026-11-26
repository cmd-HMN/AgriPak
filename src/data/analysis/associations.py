from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from src.common.dataset.columns import is_identified
from src.data.analysis.types import (
    _binary_target,
    _flt,
    _frame,
    _is_text,
    _num,
    _s,
    _txt,
)


def _bh_fdr(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked, adj, prev = arr[order], np.empty(n, dtype=float), 1.0
    for i in range(n - 1, -1, -1):
        prev = min(1.0, max(0.0, min(prev, float(ranked[i]) * n / (i + 1))))
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = adj
    return [float(v) for v in out]


def _bin_assoc(s: pd.Series, n_bins: int = 4) -> pd.Series:
    if _is_text(s):
        return _txt(s)
    num = _num(s)
    nu = int(num.dropna().nunique())
    return (
        num.fillna(-1).astype(str)
        if nu <= n_bins
        else _txt(pd.qcut(num, q=min(n_bins, nu), duplicates="drop"))
    )


def run_mann_whitney_tests(
    df: pd.DataFrame, numeric_cols: Sequence[str], target_col: str
) -> pd.DataFrame:
    cols = [
        "feature",
        "u_stat",
        "p_value",
        "p_adj_fdr",
        "median_neg",
        "median_pos",
        "rank_biserial",
    ]
    parsed = _binary_target(df, target_col)
    if parsed is None:
        return _frame([], cols)
    target, neg_v, pos_v = parsed
    rows = []
    for col in numeric_cols:
        if col not in df.columns or col == target_col:
            continue
        vals = _num(df[col])
        valid = vals.notna() & target.notna()
        pos_g = _s(vals[valid & (target == pos_v)])
        neg_g = _s(vals[valid & (target == neg_v)])
        if len(pos_g) < 2 or len(neg_g) < 2:
            continue
        mw: Any = stats.mannwhitneyu(pos_g, neg_g, alternative="two-sided")
        u, p = _flt(mw[0]), _flt(mw[1])
        rows.append(
            (
                col,
                u,
                p,
                p,
                _flt(neg_g.median()),
                _flt(pos_g.median()),
                (2.0 * u) / (len(pos_g) * len(neg_g)) - 1.0,
            )
        )
    if not rows:
        return _frame([], cols)
    out = _frame(rows, cols)
    out["p_adj_fdr"] = _bh_fdr([_flt(v) for v in out["p_value"].tolist()])
    return out.sort_values("p_value", ascending=True).reset_index(drop=True)


def compute_spearman_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    num_df = pd.DataFrame({c: _num(df[c]) for c in cols})
    valid = [c for c in num_df.columns if int(_s(num_df[c]).dropna().nunique()) > 1]
    return (
        cast(pd.DataFrame, num_df[valid]).corr(method="spearman")
        if len(valid) >= 2
        else pd.DataFrame()
    )


def compute_ranked_target_correlations(
    spearman_matrix: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    cols = ["feature", "spearman_r", "abs_spearman_r"]
    if (
        spearman_matrix.empty
        or not is_identified(target_col)
        or target_col not in spearman_matrix.columns
    ):
        return _frame([], cols)
    s = _s(spearman_matrix[target_col]).drop(index=target_col, errors="ignore").dropna()
    if s.empty:
        return _frame([], cols)
    vals = s.to_numpy(dtype=float)
    return (
        pd.DataFrame(
            {
                "feature": s.index.astype(str),
                "spearman_r": vals,
                "abs_spearman_r": np.abs(vals),
            }
        )
        .sort_values("abs_spearman_r", ascending=False)
        .reset_index(drop=True)
    )


def compute_cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if (
        len(pair) < 2
        or int(_s(pair["x"]).nunique()) < 2
        or int(_s(pair["y"]).nunique()) < 2
    ):
        return 0.0, 1.0
    ct = pd.crosstab(pair["x"], pair["y"])
    denom = int(ct.to_numpy().sum()) * min(ct.shape[0] - 1, ct.shape[1] - 1)
    if denom <= 0:
        return 0.0, 1.0
    chi: Any = stats.chi2_contingency(ct)
    return min(1.0, float(np.sqrt(max(0.0, _flt(chi[0]) / denom)))), _flt(chi[1])


def compute_cramers_v_tables(
    df: pd.DataFrame, feature_cols: Sequence[str], target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in feature_cols if c in df.columns and c != target_col]
    t_cols = ["feature", "cramers_v", "p_value"]
    if not cols:
        return _frame([], t_cols), pd.DataFrame()
    binned = {c: _bin_assoc(_s(df[c])) for c in cols}
    t_rows = (
        [
            (c, *compute_cramers_v(binned[c], _bin_assoc(_s(df[target_col]))))
            for c in cols
        ]
        if is_identified(target_col) and target_col in df.columns
        else []
    )
    t_df = (
        _frame(t_rows, t_cols)
        .sort_values("cramers_v", ascending=False)
        .reset_index(drop=True)
    )
    mat = pd.DataFrame(
        np.eye(len(cols), dtype=float), index=pd.Index(cols), columns=pd.Index(cols)
    )
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            mat.loc[c1, c2] = mat.loc[c2, c1] = compute_cramers_v(
                binned[c1], binned[c2]
            )[0]
    return t_df, mat


def run_kruskal_wallis_grid(
    df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> pd.DataFrame:
    cols = [
        "numeric_col",
        "categorical_col",
        "h_stat",
        "p_value",
        "p_adj_fdr",
        "epsilon_sq",
        "n_groups",
        "n_total",
    ]
    rows = []
    for nc in numeric_cols:
        if nc not in df.columns:
            continue
        vals = _num(df[nc])
        for cc in categorical_cols:
            if cc not in df.columns or cc == nc:
                continue
            tmp = pd.DataFrame({"val": vals, "cat": _txt(df[cc])}).dropna(
                subset=["val"]
            )
            grps = [
                g["val"].to_numpy()
                for _, g in tmp.groupby("cat", observed=True)
                if len(g) >= 2
            ]
            if len(grps) < 2:
                continue
            kw: Any = stats.kruskal(*grps)
            h, p = _flt(kw[0]), _flt(kw[1])
            nt, k = sum(len(g) for g in grps), len(grps)
            eps = min(1.0, max(0.0, (h - k + 1.0) / (nt - k))) if nt > k else 0.0
            rows.append((nc, cc, h, p, p, eps, k, nt))
    if not rows:
        return _frame([], cols)
    out = _frame(rows, cols)
    out["p_adj_fdr"] = _bh_fdr([_flt(v) for v in out["p_value"].tolist()])
    return out.sort_values("epsilon_sq", ascending=False).reset_index(drop=True)
