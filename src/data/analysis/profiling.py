from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.common.core.validation import DEFAULT_MISSING_TOKENS
from src.data.analysis.types import (
    QualityAuditResult,
    _flt,
    _frame,
    _int,
    _is_text,
    _num,
    _s,
    _target_series,
    _txt,
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
