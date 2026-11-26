from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data.analysis.types import (
    CombinationResult,
    _flt,
    _frame,
    _int,
    _num,
    _s,
    _target_series,
    _txt,
)
from src.common.dataset.columns import DatasetSchema, is_identified


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


def _score_combo_rows(
    items: list[tuple[str, str, np.ndarray]],
    y: np.ndarray,
    base: float,
    min_n: int,
    nt: int,
    ks: tuple[int, ...],
) -> tuple[list[tuple[Any, ...]], dict[str, np.ndarray]]:
    lifts = {lbl: float(np.mean(y[m])) - base for _, lbl, m in items}
    rows: list[tuple[Any, ...]] = []
    masks: dict[str, np.ndarray] = {}
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
    return rows, masks


def _replicate_combinations(
    cands: Sequence[str],
    masks: dict[str, np.ndarray],
    y: np.ndarray,
    base: float,
    min_n: int,
    random_state: int,
) -> list[tuple[Any, ...]]:
    nt = len(y)
    sa = np.zeros(nt, dtype=bool)
    sa[np.random.default_rng(random_state).permutation(nt)[: nt // 2]] = True
    sb = ~sa
    ba = float(np.mean(y[sa])) if sa.any() else base
    bb = float(np.mean(y[sb])) if sb.any() else base
    mh = max(5, min_n // 2)
    r_rows: list[tuple[Any, ...]] = []
    for name in cands:
        ma, mb = masks[name] & sa, masks[name] & sb
        na, nb = int(ma.sum()), int(mb.sum())
        ra = float(np.mean(y[ma])) if na > 0 else np.nan
        rb = float(np.mean(y[mb])) if nb > 0 else np.nan
        replicated = bool(
            na >= mh
            and nb >= mh
            and np.isfinite(ra)
            and np.isfinite(rb)
            and (ra - ba) * (rb - bb) > 0
        )
        r_rows.append((name, na, ra, nb, rb, replicated))
    return r_rows


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
    rows, masks = _score_combo_rows(items, y, base, min_n, nt, ks)
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
    cands = list(
        dict.fromkeys(risky["combination"].tolist() + prot["combination"].tolist())
    )
    rdf = _frame(
        _replicate_combinations(cands, masks, y, base, min_n, random_state), r_cols
    )
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
