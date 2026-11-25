from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.analysis.stats import compute_cramers_v, compute_numeric_shape_metrics
from src.data.analysis.types import (
    CombinationResult,
    FeatureEngineeringResult,
    QualityAuditResult,
    _binary_target,
    _flt,
    _frame,
    _get_drivers,
    _int,
    _is_text,
    _num,
    _s,
    _target_series,
    _txt,
)
from src.data.schema import DatasetSchema
from src.data.utils import UNIDENTIFIED, is_identified


def _single_feature_metrics(
    col: str, s: pd.Series, yv: pd.Series, splits: int, random_state: int
) -> tuple[float, float]:
    if _is_text(s):
        ts = _txt(s)
        corr, _ = compute_cramers_v(ts, yv)
        auc = 0.5
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
            cv = StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=random_state
            )
            auc = float(
                np.mean(
                    cross_val_score(
                        pipe, pd.DataFrame({col: ts}), yv, cv=cv, scoring="roc_auc"
                    )
                )
            )
        return corr, auc
    pair = pd.DataFrame({"x": _num(s), "y": yv}).dropna()
    if (
        len(pair) < 4
        or int(_s(pair["x"]).nunique()) < 2
        or int(_s(pair["y"]).nunique()) != 2
    ):
        return 0.0, 0.5
    r = _s(pair["x"]).corr(_s(pair["y"]), method="spearman")
    corr = abs(_flt(r)) if pd.notna(r) else 0.0
    ra = _flt(roc_auc_score(pair["y"], pair["x"]))
    return corr, max(ra, 1.0 - ra)


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
    splits = min(n_splits, _int(yv.value_counts().min()))
    id_cols = set(getattr(schema, "id_cols", ()))
    rows, flagged = [], []
    for col in df.columns:
        if col == schema.target_column or col in id_cols:
            continue
        corr, auc = _single_feature_metrics(
            col, _s(df.loc[vmask, col]), yv, splits, random_state
        )
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
    out = _frame(rows, cols).sort_values(
        ["flagged", "single_feature_auc", "abs_spearman"],
        ascending=[False, False, False],
    )
    return out.reset_index(drop=True), tuple(flagged)


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
        val = (
            safe.iloc[0]["min_val"]
            if direction == "below"
            else safe.iloc[-1]["max_val"]
        )
        return _flt(val), direction
    return _flt(_s(v["x"]).median()), direction


def _feature_correlation_table(
    out: pd.DataFrame, c_cols: Sequence[str], eng_t: tuple[str, ...], target_col: str
) -> pd.DataFrame:
    c_rows, target = [], _target_series(out, target_col)
    if target is not None:
        for col in dict.fromkeys(c_cols):
            pair = pd.DataFrame({"x": _num(out[col]), "y": target}).dropna()
            if len(pair) >= 3 and int(_s(pair["x"]).nunique()) > 1:
                r = _flt(_s(pair["x"]).corr(_s(pair["y"]), method="spearman"))
                c_rows.append((col, col in eng_t, r, abs(r)))
    c_df = _frame(c_rows, ["feature", "is_engineered", "spearman_r", "abs_spearman_r"])
    return (
        c_df.sort_values("abs_spearman_r", ascending=False).reset_index(drop=True)
        if not c_df.empty
        else c_df
    )


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
    return FeatureEngineeringResult(
        out,
        eng_t,
        raw_sig,
        thresh,
        direction,
        compute_numeric_shape_metrics(out, eng_t),
        _feature_correlation_table(out, c_cols, eng_t, schema.target_column),
    )


def _evaluate_feature_set_cv(
    df: pd.DataFrame,
    y: pd.Series,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    n_splits: int,
    random_state: int,
) -> tuple[float, float, float, float] | None:
    un = [c for c in num_cols if c in df.columns]
    uc = [c for c in cat_cols if c in df.columns]
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
