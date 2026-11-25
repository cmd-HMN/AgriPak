from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from src.data.analysis.types import (
    QualityAuditResult,
    _binary_target,
    _flt,
    _frame,
    _int,
    _is_text,
    _num,
    _s,
    _target_series,
    _txt,
)
from src.data.utils import DEFAULT_MISSING_TOKENS, is_identified


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


def _text_hygiene_rows(
    src: pd.DataFrame, tokens: set[str]
) -> list[tuple[str, int, int, int]]:
    rows: list[tuple[str, int, int, int]] = []
    for c in src.columns:
        s = _s(src[c])
        if not _is_text(s):
            continue
        nn = s.dropna().astype(str)
        if nn.empty:
            rows.append((str(c), 0, 0, 0))
            continue
        st, lo = nn.str.strip(), nn.str.strip().str.lower()
        rows.append(
            (
                str(c),
                _int((nn != st).sum()),
                _int(lo.isin(tokens).sum()),
                max(0, _int(st.nunique()) - _int(lo.nunique())),
            )
        )
    return rows


def audit_data_quality(
    df: pd.DataFrame,
    raw_text_df: pd.DataFrame | None = None,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> QualityAuditResult:
    s_cols = ["column", "dtype", "non_null", "missing_count", "missing_pct", "n_unique"]
    h_cols = ["column", "whitespace_issues", "missing_token_hits", "case_variants"]
    if df.empty or len(df.columns) == 0:
        return QualityAuditResult(_frame([], s_cols), 0, _frame([], h_cols), 0.0)
    n = len(df)
    s_rows = [
        (
            str(c),
            str(s.dtype),
            n - (m := _int(s.isna().sum())),
            m,
            m / n * 100.0,
            _int(s.nunique(dropna=True)),
        )
        for c in df.columns
        for s in (_s(df[c]),)
    ]
    src = raw_text_df if raw_text_df is not None else df
    tokens = {str(t).strip().lower() for t in missing_tokens}
    return QualityAuditResult(
        _frame(s_rows, s_cols),
        _int(df.duplicated().sum()),
        _frame(_text_hygiene_rows(src, tokens), h_cols),
        _flt(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0),
    )


def compute_numeric_shape_metrics(
    df: pd.DataFrame, numeric_cols: Sequence[str]
) -> pd.DataFrame:
    cols = [
        "column",
        "count",
        "mean",
        "std",
        "median",
        "iqr",
        "min",
        "max",
        "skewness",
        "kurtosis",
        "outlier_count_iqr",
        "outlier_pct",
    ]
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = _num(df[col]).dropna()
        n = len(s)
        if n == 0:
            continue
        q25, q50, q75 = (
            _flt(s.quantile(0.25)),
            _flt(s.quantile(0.50)),
            _flt(s.quantile(0.75)),
        )
        iqr, sd = q75 - q25, _flt(s.std(ddof=1)) if n > 1 else 0.0
        out_n = _int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        sk = _flt(s.skew()) if n >= 3 and sd > 0 else 0.0
        ku = _flt(s.kurtosis()) if n >= 4 and sd > 0 else 0.0
        rows.append(
            (
                col,
                n,
                _flt(s.mean()),
                sd,
                q50,
                iqr,
                _flt(s.min()),
                _flt(s.max()),
                sk,
                ku,
                out_n,
                out_n / n * 100.0,
            )
        )
    return _frame(rows, cols)


def compute_categorical_summaries(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, pd.DataFrame]:
    n, res = len(df), {}
    for col in columns:
        if col in df.columns:
            cnt = _txt(df[col]).value_counts(dropna=False)
            vals = cnt.to_numpy(dtype=float)
            res[col] = pd.DataFrame(
                {
                    "category": cnt.index.astype(str),
                    "count": cnt.to_numpy(dtype=int),
                    "pct": vals / n * 100.0 if n > 0 else np.zeros(len(vals)),
                }
            ).reset_index(drop=True)
    return res


def compute_target_rates_by_group(
    df: pd.DataFrame, group_cols: Sequence[str], target_col: str
) -> dict[str, pd.DataFrame]:
    target = _target_series(df, target_col)
    if target is None:
        return {}
    res: dict[str, pd.DataFrame] = {}
    for col in group_cols:
        if col not in df.columns or col == target_col:
            continue
        tmp = pd.DataFrame({"group": _txt(df[col]), "target": target}).dropna(
            subset=["target"]
        )
        if tmp.empty:
            continue
        grp = cast(
            pd.DataFrame,
            tmp.groupby("group", observed=True)["target"]
            .agg(count="count", positive_count="sum", target_rate="mean", std="std")
            .reset_index()
            .sort_values("target_rate", ascending=False)
            .reset_index(drop=True),
        )
        grp["std"] = _s(grp["std"]).fillna(0.0)
        res[col] = grp
    return res


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


def compute_decile_target_curves(
    df: pd.DataFrame, numeric_cols: Sequence[str], target_col: str, n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    target = _target_series(df, target_col)
    if target is None:
        return {}
    curves: dict[str, pd.DataFrame] = {}
    for col in numeric_cols:
        if col not in df.columns or col == target_col:
            continue
        sub = pd.DataFrame({"val": _num(df[col]), "target": target}).dropna()
        nu = int(_s(sub["val"]).nunique())
        if len(sub) < 4 or nu < 2:
            continue
        bins = _s(pd.qcut(sub["val"], q=min(n_bins, nu), duplicates="drop"))
        if int(bins.nunique()) < 2:
            continue
        sub["bin"] = bins
        grp = cast(
            pd.DataFrame,
            sub.groupby("bin", observed=True)
            .agg(
                count=("target", "count"),
                mean_value=("val", "mean"),
                min_value=("val", "min"),
                max_value=("val", "max"),
                target_rate=("target", "mean"),
            )
            .reset_index(),
        )
        grp["bin_index"] = np.arange(len(grp), dtype=int)
        grp["bin_label"] = grp["bin"].astype(str)
        curves[col] = grp.drop(columns=["bin"])
    return curves


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
