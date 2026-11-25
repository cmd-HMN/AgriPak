from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.errors import AnalysisError
from src.data.schema import DatasetSchema, resolve_drivers
from src.data.utils import (
    DEFAULT_MISSING_TOKENS,
    UNIDENTIFIED,
    compute_min_subgroup_size,
    is_identified,
)


@dataclass(frozen=True)
class QualityAuditResult:
    column_summary: pd.DataFrame
    duplicate_count: int
    text_hygiene: pd.DataFrame
    memory_mb: float


@dataclass(frozen=True)
class CombinationResult:
    total_combos: int
    top_risky: pd.DataFrame
    top_protective: pd.DataFrame
    top_synergy: pd.DataFrame
    replication_table: pd.DataFrame
    replication_rate: float


@dataclass(frozen=True)
class FeatureEngineeringResult:
    data: pd.DataFrame
    engineered_columns: tuple[str, ...]
    raw_signal_columns: tuple[str, ...]
    inferred_threshold: float | None
    threshold_direction: str
    summary_stats: pd.DataFrame
    correlation_comparison: pd.DataFrame


@dataclass(frozen=True)
class AnalysisBundle:
    schema: DatasetSchema
    row_count: int
    col_count: int
    base_rate: float | None
    min_n: int
    quality_audit: QualityAuditResult
    numeric_describe: pd.DataFrame
    numeric_shape: pd.DataFrame
    categorical_counts: dict[str, pd.DataFrame]
    target_rates_by_group: dict[str, pd.DataFrame]
    mann_whitney: pd.DataFrame
    spearman_matrix: pd.DataFrame
    ranked_target_spearman: pd.DataFrame
    decile_curves: dict[str, pd.DataFrame]
    combinations: CombinationResult
    subgroup_surfaces: dict[str, pd.DataFrame]
    kruskal_wallis: pd.DataFrame
    target_cramers_v: pd.DataFrame
    pairwise_cramers_v: pd.DataFrame
    leakage_table: pd.DataFrame
    flagged_leakage_columns: tuple[str, ...]
    synthetic_notes: tuple[str, ...]
    feature_engineering: FeatureEngineeringResult
    cv_auc_table: pd.DataFrame


def _s(v: Any) -> pd.Series:
    return cast(pd.Series, v)


def _num(v: Any) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(v, errors="coerce"))


def _flt(v: Any) -> float:
    return float(cast(Any, v))


def _int(v: Any) -> int:
    return int(cast(Any, v))


def _txt(v: Any) -> pd.Series:
    return _s(v).astype("object").fillna(UNIDENTIFIED).astype(str)


def _frame(rows: Sequence[Any], cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(cols))


def _is_text(s: pd.Series) -> bool:
    return bool(
        pd.api.types.is_string_dtype(s)
        or s.dtype == object
        or isinstance(s.dtype, pd.CategoricalDtype)
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


def _get_drivers(
    schema: DatasetSchema, df: pd.DataFrame | None = None
) -> tuple[str, str]:
    if is_identified(schema.primary_driver) and is_identified(schema.secondary_driver):
        return schema.primary_driver, schema.secondary_driver
    if df is not None:
        r = resolve_drivers(df, schema)
        if is_identified(r.primary_driver) or is_identified(r.secondary_driver):
            return r.primary_driver, r.secondary_driver
    nc = schema.numeric_columns
    return nc[0] if nc else UNIDENTIFIED, nc[1] if len(nc) > 1 else UNIDENTIFIED


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


def _target_series(df: pd.DataFrame, target_col: str) -> pd.Series | None:
    if not is_identified(target_col) or target_col not in df.columns:
        return None
    t = _num(df[target_col])
    return t if int(t.dropna().nunique()) >= 2 else None


def _binary_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.Series, float, float] | None:
    t = _target_series(df, target_col)
    if t is None:
        return None
    u = sorted(_flt(v) for v in t.dropna().unique())
    return (t, u[0], u[1]) if len(u) == 2 else None


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
    h_rows = []
    for c in src.columns:
        s = _s(src[c])
        if not _is_text(s):
            continue
        nn = s.dropna().astype(str)
        if nn.empty:
            h_rows.append((str(c), 0, 0, 0))
            continue
        st, lo = nn.str.strip(), nn.str.strip().str.lower()
        h_rows.append(
            (
                str(c),
                _int((nn != st).sum()),
                _int(lo.isin(tokens).sum()),
                max(0, _int(st.nunique()) - _int(lo.nunique())),
            )
        )
    return QualityAuditResult(
        _frame(s_rows, s_cols),
        _int(df.duplicated().sum()),
        _frame(h_rows, h_cols),
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
        sk, ku = (
            (_flt(s.skew()) if n >= 3 and sd > 0 else 0.0),
            (_flt(s.kurtosis()) if n >= 4 and sd > 0 else 0.0),
        )
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
        pos_g, neg_g = (
            _s(vals[valid & (target == pos_v)]),
            _s(vals[valid & (target == neg_v)]),
        )
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
        grp["bin_index"], grp["bin_label"] = (
            np.arange(len(grp), dtype=int),
            grp["bin"].astype(str),
        )
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


def _build_condition_items(
    df: pd.DataFrame, schema: DatasetSchema, min_n: int
) -> list[tuple[str, str, np.ndarray]]:
    items: list[tuple[str, str, np.ndarray]] = []
    tc = schema.target_column
    for col in schema.binary_columns:
        if col in df.columns and col != tc:
            v = _num(df[col])
            for fv in (1, 0):
                if int((m := (v == fv).to_numpy(dtype=bool)).sum()) >= min_n:
                    items.append((col, f"{col}={fv}", m))
    for col in schema.categorical:
        if col in df.columns and col != tc:
            s = _txt(df[col])
            for cv, cnt in s.value_counts().items():
                if _int(cnt) >= min_n:
                    items.append(
                        (col, f"{col}={cv}", (s == str(cv)).to_numpy(dtype=bool))
                    )
    for col in schema.numeric_columns:
        if col in df.columns and col != tc:
            v = _num(df[col])
            valid = v.dropna()
            if int(valid.nunique()) < 3:
                continue
            ql, qh = _flt(valid.quantile(0.25)), _flt(valid.quantile(0.75))
            if int((lm := (v <= ql).to_numpy(dtype=bool)).sum()) >= min_n:
                items.append((col, f"{col}<=Q25", lm))
            if qh > ql and int((hm := (v >= qh).to_numpy(dtype=bool)).sum()) >= min_n:
                items.append((col, f"{col}>=Q75", hm))
    return items


def analyze_feature_combinations(
    df: pd.DataFrame,
    schema: DatasetSchema,
    min_n: int,
    random_state: int = 42,
    ks: tuple[int, ...] = (2, 3),
    top_n: int = 15,
) -> CombinationResult:
    c_cols = ["combination", "k", "n", "target_rate", "lift", "synergy"]
    r_cols = [
        "combination",
        "n_split_a",
        "rate_split_a",
        "n_split_b",
        "rate_split_b",
        "replicated",
    ]
    empty = CombinationResult(
        0,
        _frame([], c_cols),
        _frame([], c_cols),
        _frame([], c_cols),
        _frame([], r_cols),
        0.0,
    )
    target = _target_series(df, schema.target_column)
    if target is None or df.empty:
        return empty
    v_idx = target.notna()
    wdf = cast(pd.DataFrame, df.loc[v_idx]).reset_index(drop=True)
    y = _s(target.loc[v_idx]).to_numpy(dtype=float)
    nt, base = len(wdf), float(np.mean(y))
    items = _build_condition_items(wdf, schema, min_n)
    if not items:
        return empty

    lifts = {lbl: float(np.mean(y[m])) - base for _, lbl, m in items}
    rows, masks = [], {}
    for k in ks:
        for grp in combinations(items, k):
            if len({g[0] for g in grp}) < k:
                continue
            cm = np.logical_and.reduce([g[2] for g in grp])
            nm = int(cm.sum())
            if nm < min_n or nm >= nt:
                continue
            rate = float(np.mean(y[cm]))
            lift = rate - base
            lbl = " & ".join(g[1] for g in grp)
            masks[lbl] = cm
            rows.append((lbl, k, nm, rate, lift, lift - sum(lifts[g[1]] for g in grp)))
    if not rows:
        return empty

    adf = _frame(rows, c_cols)
    risky = (
        cast(pd.DataFrame, adf[adf["lift"] > 0])
        .sort_values(["target_rate", "n"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    prot = (
        cast(pd.DataFrame, adf[adf["lift"] < 0])
        .sort_values(["target_rate", "n"], ascending=[True, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    syn = (
        adf.sort_values(["synergy", "n"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    sa = np.zeros(nt, dtype=bool)
    sa[np.random.default_rng(random_state).permutation(nt)[: nt // 2]] = True
    sb = ~sa
    ba, bb, mh = (
        (float(np.mean(y[sa])) if sa.any() else base),
        (float(np.mean(y[sb])) if sb.any() else base),
        max(5, min_n // 2),
    )
    cands = list(
        dict.fromkeys(risky["combination"].tolist() + prot["combination"].tolist())
    )
    r_rows = []
    for name in cands:
        ma, mb = masks[name] & sa, masks[name] & sb
        na, nb = int(ma.sum()), int(mb.sum())
        ra, rb = (
            (float(np.mean(y[ma])) if na > 0 else np.nan),
            (float(np.mean(y[mb])) if nb > 0 else np.nan),
        )
        r_rows.append(
            (
                name,
                na,
                ra,
                nb,
                rb,
                bool(
                    na >= mh
                    and nb >= mh
                    and np.isfinite(ra)
                    and np.isfinite(rb)
                    and (ra - ba) * (rb - bb) > 0
                ),
            )
        )
    rdf = _frame(r_rows, r_cols)
    return CombinationResult(
        len(adf),
        risky,
        prot,
        syn,
        rdf,
        _flt(rdf["replicated"].mean()) if not rdf.empty else 0.0,
    )


def compute_subgroup_risk_surfaces(
    df: pd.DataFrame,
    driver_x: str,
    driver_y: str,
    target_col: str,
    subgroup_col: str,
    min_n: int,
    n_bins: int = 4,
) -> dict[str, pd.DataFrame]:
    if driver_x == driver_y or any(
        not is_identified(c) or c not in df.columns
        for c in (driver_x, driver_y, target_col, subgroup_col)
    ):
        return {}
    w = pd.DataFrame(
        {
            "x": _num(df[driver_x]),
            "y": _num(df[driver_y]),
            "t": _num(df[target_col]),
            "sub": _txt(df[subgroup_col]),
        }
    ).dropna(subset=["x", "y", "t"])
    nx, ny = int(_s(w["x"]).nunique()), int(_s(w["y"]).nunique())
    if len(w) < min_n or nx < 2 or ny < 2:
        return {}
    w["x_bin"] = _s(pd.qcut(w["x"], q=min(n_bins, nx), duplicates="drop")).astype(str)
    w["y_bin"] = _s(pd.qcut(w["y"], q=min(n_bins, ny), duplicates="drop")).astype(str)
    surfaces, cmin = {}, max(3, min_n // 5)
    for grp, gdf in w.groupby("sub", observed=True):
        if len(gdf) < min_n:
            continue
        agg = cast(
            pd.DataFrame,
            gdf.groupby(["y_bin", "x_bin"], observed=True)["t"]
            .agg(count="count", rate="mean")
            .reset_index(),
        )
        agg.loc[agg["count"] < cmin, "rate"] = np.nan
        piv = agg.pivot(index="y_bin", columns="x_bin", values="rate")
        if not piv.empty:
            surfaces[str(grp)] = piv
    return surfaces


def run_leakage_check(
    df: pd.DataFrame,
    schema: DatasetSchema,
    leak_threshold: float = 0.90,
    auc_threshold: float = 0.95,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    cols = ["feature", "abs_spearman", "single_feature_auc", "flagged", "reason"]
    parsed = _binary_target(df, schema.target_column)
    if parsed is None:
        return _frame([], cols), ()
    target, _, pos_v = parsed
    vmask = target.notna()
    yv = (_s(target.loc[vmask]) == pos_v).astype(int)
    splits, id_cols, rows, flagged = (
        min(n_splits, _int(yv.value_counts().min())),
        set(getattr(schema, "id_cols", ())),
        [],
        [],
    )

    for col in df.columns:
        if col == schema.target_column or col in id_cols:
            continue
        s = _s(df.loc[vmask, col])
        corr, auc = 0.0, 0.5
        if _is_text(s):
            ts = _txt(s)
            corr, _ = compute_cramers_v(ts, yv)
            if splits >= 2 and int(s.nunique(dropna=True)) > 1:
                pipe = Pipeline(
                    [
                        ("enc", OneHotEncoder(handle_unknown="ignore")),
                        (
                            "clf",
                            LogisticRegression(max_iter=500, random_state=random_state),
                        ),
                    ]
                )
                auc = float(
                    np.mean(
                        cross_val_score(
                            pipe,
                            pd.DataFrame({col: ts}),
                            yv,
                            cv=StratifiedKFold(
                                n_splits=splits, shuffle=True, random_state=random_state
                            ),
                            scoring="roc_auc",
                        )
                    )
                )
        else:
            pair = pd.DataFrame({"x": _num(s), "y": yv}).dropna()
            if (
                len(pair) >= 4
                and int(_s(pair["x"]).nunique()) >= 2
                and int(_s(pair["y"]).nunique()) == 2
            ):
                r = _s(pair["x"]).corr(_s(pair["y"]), method="spearman")
                corr = abs(_flt(r)) if pd.notna(r) else 0.0
                ra = _flt(roc_auc_score(pair["y"], pair["x"]))
                auc = max(ra, 1.0 - ra)
        reasons = [
            m
            for ok, m in (
                (corr >= leak_threshold, f"corr>={leak_threshold:.2f}"),
                (auc >= auc_threshold, f"auc>={auc_threshold:.2f}"),
            )
            if ok
        ]
        if reasons:
            flagged.append(col)
        rows.append(
            (col, corr, auc, bool(reasons), ", ".join(reasons) if reasons else "clean")
        )
    return _frame(rows, cols).sort_values(
        ["flagged", "single_feature_auc", "abs_spearman"],
        ascending=[False, False, False],
    ).reset_index(drop=True), tuple(flagged)


def infer_risk_floor_threshold(
    df: pd.DataFrame,
    driver_col: str,
    target_col: str,
    n_bins: int = 10,
    near_zero_rate: float = 0.05,
) -> tuple[float | None, str]:
    if (
        not is_identified(driver_col)
        or not is_identified(target_col)
        or driver_col not in df.columns
        or target_col not in df.columns
    ):
        return None, "none"
    v = pd.DataFrame({"x": _num(df[driver_col]), "y": _num(df[target_col])}).dropna()
    nx = int(_s(v["x"]).nunique())
    if len(v) < 10 or nx < 2 or int(_s(v["y"]).nunique()) < 2:
        return None, "none"
    direction = (
        "below"
        if _flt(_s(v["x"]).corr(_s(v["y"]), method="spearman")) <= 0
        else "above"
    )
    v["bin"] = _s(pd.qcut(v["x"], q=min(n_bins, nx), duplicates="drop"))
    grp = cast(
        pd.DataFrame,
        v.groupby("bin", observed=True)
        .agg(min_val=("x", "min"), max_val=("x", "max"), rate=("y", "mean"))
        .reset_index(),
    )
    if grp.empty:
        return _flt(_s(v["x"]).median()), direction
    cutoff = max(near_zero_rate, _flt(_s(v["y"]).mean()) * 0.5)
    safe = cast(pd.DataFrame, grp[grp["rate"] <= near_zero_rate])
    if safe.empty:
        safe = cast(pd.DataFrame, grp[grp["rate"] <= cutoff])
    if not safe.empty:
        return _flt(
            safe.iloc[0]["min_val"]
            if direction == "below"
            else safe.iloc[-1]["max_val"]
        ), direction
    return _flt(_s(v["x"]).median()), direction


def engineer_features(
    df: pd.DataFrame, schema: DatasetSchema, threshold: float | None = None
) -> FeatureEngineeringResult:
    out = df.copy()
    raw_sig = tuple(
        c
        for c in schema.binary_columns
        if c in out.columns and c != schema.target_column
    )
    primary, _ = _get_drivers(schema, df=out)
    thresh, direction = threshold, "below"
    if is_identified(primary) and is_identified(schema.target_column):
        auto_t, direction = infer_risk_floor_threshold(
            out, primary, schema.target_column
        )
        thresh = auto_t if thresh is None else thresh

    eng: list[str] = []
    if raw_sig:
        out["binary_signal_sum"] = pd.DataFrame(
            {c: _num(out[c]).fillna(0.0) for c in raw_sig}, index=out.index
        ).sum(axis=1)
        eng.append("binary_signal_sum")
    if is_identified(primary) and primary in out.columns and thresh is not None:
        dv = _num(out[primary])
        deficit = np.maximum(0.0, dv - thresh if direction == "above" else thresh - dv)
        out["primary_driver_deficit"] = deficit
        eng.append("primary_driver_deficit")
        if "binary_signal_sum" in out.columns:
            out["deficit_x_zero_signal"] = deficit * (
                _s(out["binary_signal_sum"]) == 0
            ).astype(float)
            fv = _flt(dv.median()) if bool(dv.notna().any()) else 0.0
            out["primary_x_binary_sum"] = dv.fillna(fv) * _s(out["binary_signal_sum"])
            eng.extend(["deficit_x_zero_signal", "primary_x_binary_sum"])

    eng_t = tuple(eng)
    c_cols = (
        list(raw_sig)
        + ([primary] if is_identified(primary) and primary in out.columns else [])
        + list(eng_t)
    )
    c_rows, target = [], _target_series(out, schema.target_column)
    if target is not None:
        for col in dict.fromkeys(c_cols):
            pair = pd.DataFrame({"x": _num(out[col]), "y": target}).dropna()
            if len(pair) >= 3 and int(_s(pair["x"]).nunique()) > 1:
                r = _flt(_s(pair["x"]).corr(_s(pair["y"]), method="spearman"))
                c_rows.append((col, col in eng_t, r, abs(r)))
    c_df = _frame(c_rows, ["feature", "is_engineered", "spearman_r", "abs_spearman_r"])
    if not c_df.empty:
        c_df = c_df.sort_values("abs_spearman_r", ascending=False).reset_index(
            drop=True
        )
    return FeatureEngineeringResult(
        out,
        eng_t,
        raw_sig,
        thresh,
        direction,
        compute_numeric_shape_metrics(out, eng_t),
        c_df,
    )


def _evaluate_feature_set_cv(
    df: pd.DataFrame,
    y: pd.Series,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    n_splits: int,
    random_state: int,
) -> tuple[float, float, float, float] | None:
    un, uc = (
        [c for c in num_cols if c in df.columns],
        [c for c in cat_cols if c in df.columns],
    )
    if not un and not uc:
        return None
    tr: list[tuple[str, Any, list[str]]] = []
    if un:
        tr.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                un,
            )
        )
    if uc:
        tr.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imp",
                            SimpleImputer(strategy="constant", fill_value=UNIDENTIFIED),
                        ),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                uc,
            )
        )
    pipe = Pipeline(
        [
            ("prep", ColumnTransformer(tr, remainder="drop")),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    sc = cross_val_score(
        pipe,
        df[un + uc],
        y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="roc_auc",
    )
    return float(np.mean(sc)), float(np.std(sc)), float(np.min(sc)), float(np.max(sc))


def validate_models_cv(
    df: pd.DataFrame,
    schema: DatasetSchema,
    fe_result: FeatureEngineeringResult,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    cols = ["model_set", "n_features", "mean_auc", "std_auc", "min_auc", "max_auc"]
    parsed = _binary_target(df, schema.target_column)
    if parsed is None:
        return _frame([], cols)
    target, _, pos_v = parsed
    vmask = target.notna()
    wdf = cast(pd.DataFrame, fe_result.data.loc[vmask]).copy()
    y = (_s(target.loc[vmask]) == pos_v).astype(int)
    splits = min(n_splits, _int(y.value_counts().min()))
    if splits < 2:
        return _frame([], cols)
    tc = schema.target_column
    raw_num = list(
        dict.fromkeys(
            c
            for c in (*schema.numeric_columns, *schema.binary_columns)
            if c in wdf.columns and c != tc
        )
    )
    raw_cat = [c for c in schema.categorical if c in wdf.columns and c != tc]
    eng = [c for c in fe_result.engineered_columns if c in wdf.columns]
    for col in raw_cat:
        wdf[col] = _txt(wdf[col])
    for col in raw_num + eng:
        wdf[col] = _num(wdf[col])
    cfgs: list[tuple[str, list[str], list[str]]] = [("raw_features", raw_num, raw_cat)]
    if eng:
        cfgs.extend(
            [
                ("engineered_only", eng, []),
                ("raw_plus_engineered", list(dict.fromkeys(raw_num + eng)), raw_cat),
            ]
        )
    rows = [
        (name, len(nc) + len(cc), *m)
        for name, nc, cc in cfgs
        if (m := _evaluate_feature_set_cv(wdf, y, nc, cc, splits, random_state))
        is not None
    ]
    return _frame(rows, cols)


def _collect_synthetic_notes(
    qa: QualityAuditResult,
    shape: pd.DataFrame,
    flagged: tuple[str, ...],
    combos: CombinationResult,
) -> tuple[str, ...]:
    notes: list[str] = []
    if qa.duplicate_count > 0:
        notes.append(f"Detected {qa.duplicate_count} exact duplicate rows.")
    if flagged:
        notes.append(f"Potential target leakage columns: {', '.join(flagged)}.")
    if not shape.empty and bool((shape["std"] == 0).any()):
        zc = [str(c) for c in _s(shape.loc[shape["std"] == 0, "column"]).tolist()]
        notes.append(f"Zero-variance numeric columns: {', '.join(zc)}.")
    if combos.total_combos > 0 and combos.replication_rate < 0.5:
        notes.append(
            f"Low combination split-half replication rate ({combos.replication_rate:.1%})."
        )
    return tuple(notes)


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
