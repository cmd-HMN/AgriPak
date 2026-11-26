from dataclasses import dataclass

import pandas as pd

from src.common.core.validation import (
    as_series,
    require_non_empty,
    to_numeric_series,
)
from src.common.dataset.columns import (
    DatasetSchema,
    extract_binary_target,
    infer_schema,
    series_to_float,
    series_to_text,
)
from src.common.dataset.stats import compute_cramers_v, compute_spearman_with_target


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_features: list[str]
    dropped_features: list[str]
    selected_numeric: list[str]
    selected_binary: list[str]
    selected_categorical: list[str]
    selected_engineered: list[str]
    manifest: pd.DataFrame


def _score_numeric_or_binary(
    df: pd.DataFrame, col: str, target: pd.Series, is_binary: bool
) -> float:
    s = as_series(df, col)
    series = series_to_float(s) if is_binary else to_numeric_series(s)
    return (
        float(compute_spearman_with_target(series, target, target))
        if False
        else float(series.corr(target, method="spearman") or 0.0)
    )


def _score_categorical(df: pd.DataFrame, col: str, target: pd.Series) -> float:
    cat = series_to_text(as_series(df, col))
    return compute_cramers_v(cat, target)


def _manifest_row(
    feature: str, decision: str, group: str, metric: str, value: float, reason: str
) -> dict[str, object]:
    return {
        "feature": feature,
        "decision": decision,
        "group": group,
        "signal_metric": metric,
        "signal_value": round(float(value), 4),
        "reason": reason,
    }


def _evaluate_column(
    df: pd.DataFrame,
    col: str,
    group: str,
    target: pd.Series,
    min_spearman: float,
    min_cramers_v: float,
) -> tuple[bool, dict[str, object]]:
    if group == "categorical":
        score = _score_categorical(df, col, target)
        keep = score >= min_cramers_v
        reason = (
            f"cramers_v={score:.4f} >= {min_cramers_v:.2f}"
            if keep
            else f"noise: cramers_v={score:.4f} < {min_cramers_v:.2f}"
        )
        return keep, _manifest_row(
            col, "keep" if keep else "drop", group, "cramers_v", score, reason
        )
    score = _score_numeric_or_binary(df, col, target, is_binary=(group == "binary"))
    keep = abs(score) >= min_spearman
    reason = (
        f"|spearman|={abs(score):.4f} >= {min_spearman:.2f}"
        if keep
        else f"noise: |spearman|={abs(score):.4f} < {min_spearman:.2f}"
    )
    return keep, _manifest_row(
        col, "keep" if keep else "drop", group, "spearman", score, reason
    )


def _score_group(
    df: pd.DataFrame,
    cols: list[str],
    eng_cols: list[str],
    group: str,
    target: pd.Series,
    min_spearman: float,
    min_cramers_v: float,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    kept: list[str] = []
    dropped: list[str] = []
    rows: list[dict[str, object]] = []
    for col in cols:
        if col in df.columns and col not in eng_cols:
            keep, row = _evaluate_column(
                df, col, group, target, min_spearman, min_cramers_v
            )
            rows.append(row)
            (kept if keep else dropped).append(col)
    return kept, dropped, rows


def _fallback_no_target(
    df: pd.DataFrame, active: DatasetSchema, eng_cols: list[str]
) -> FeatureSelectionResult:
    num = [c for c in active.numeric_columns if c in df.columns]
    bin_c = [c for c in active.binary_columns if c in df.columns]
    cat = [c for c in active.categorical if c in df.columns]
    raw_keep = num + bin_c + cat
    rows = [
        _manifest_row(c, "keep", "raw", "none", 0.0, "no_target_fallback")
        for c in raw_keep + eng_cols
    ]
    return FeatureSelectionResult(
        raw_keep + eng_cols, [], num, bin_c, cat, eng_cols, pd.DataFrame(rows)
    )


def select_features(
    df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    engineered_cols: list[str] | None = None,
    min_spearman: float = 0.03,
    min_cramers_v: float = 0.05,
) -> FeatureSelectionResult:
    require_non_empty(df, "df")
    active = schema or infer_schema(df)
    eng_cols = [c for c in (engineered_cols or []) if c in df.columns]
    if not active.target_column or active.target_column not in df.columns:
        return _fallback_no_target(df, active, eng_cols)
    target = extract_binary_target(df, active.target_column)
    sel_num, drop_num, r_num = _score_group(
        df,
        list(active.numeric_columns),
        eng_cols,
        "numeric",
        target,
        min_spearman,
        min_cramers_v,
    )
    sel_bin, drop_bin, r_bin = _score_group(
        df,
        list(active.binary_columns),
        eng_cols,
        "binary",
        target,
        min_spearman,
        min_cramers_v,
    )
    sel_cat, drop_cat, r_cat = _score_group(
        df,
        list(active.categorical),
        eng_cols,
        "categorical",
        target,
        min_spearman,
        min_cramers_v,
    )
    r_eng = [
        _manifest_row(
            c,
            "keep",
            "engineered",
            "spearman",
            _score_numeric_or_binary(df, c, target, False),
            "domain_engineered_feature",
        )
        for c in eng_cols
    ]
    r_unid = [
        _manifest_row(
            c,
            "drop",
            "unidentified",
            "constant_or_unidentified",
            0.0,
            "noise: constant or unidentified",
        )
        for c in active.unidentified_columns
        if c in df.columns and c not in eng_cols and c != active.target_column
    ]
    drop_unid = [str(r["feature"]) for r in r_unid]
    return FeatureSelectionResult(
        selected_features=sel_num + sel_bin + sel_cat + eng_cols,
        dropped_features=drop_num + drop_bin + drop_cat + drop_unid,
        selected_numeric=sel_num,
        selected_binary=sel_bin,
        selected_categorical=sel_cat,
        selected_engineered=eng_cols,
        manifest=pd.DataFrame(r_num + r_bin + r_cat + r_unid + r_eng),
    )
