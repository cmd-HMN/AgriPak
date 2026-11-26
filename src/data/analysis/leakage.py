from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.dataset.columns import DatasetSchema, is_identified
from src.data.analysis.associations import compute_cramers_v
from src.data.analysis.types import (
    CombinationResult,
    QualityAuditResult,
    _binary_target,
    _frame,
    _num,
    _s,
)


def _score_numeric_feature(
    s: pd.Series, y_num: pd.Series, valid_y: pd.Series
) -> tuple[str, float]:
    vals = _num(s)
    mask = valid_y & vals.notna()
    if int(mask.sum()) < 20 or int(vals.loc[mask].nunique()) < 2:
        return ("roc_auc", 0.5)
    y_sub = y_num.loc[mask].astype(int)
    if int(y_sub.nunique()) < 2:
        return ("roc_auc", 0.5)
    auc = float(cast(Any, roc_auc_score(y_sub, vals.loc[mask])))
    return ("roc_auc", max(auc, 1.0 - auc))


def _score_categorical_feature(
    s: pd.Series, y_num: pd.Series, valid_y: pd.Series
) -> tuple[str, float]:
    mask = valid_y & s.notna()
    if int(mask.sum()) < 20:
        return ("cramers_v", 0.0)
    v, _ = compute_cramers_v(_s(s.loc[mask]), _s(y_num.loc[mask]))
    return ("cramers_v", float(v))


def run_leakage_check(
    df: pd.DataFrame,
    schema: DatasetSchema,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    cols = ["feature", "metric", "score", "flagged"]
    tc = schema.target_column
    if df.empty or not is_identified(tc) or tc not in df.columns:
        return _frame([], cols), ()
    res = _binary_target(df, tc)
    if res is None:
        return _frame([], cols), ()
    y_num = res[0]
    valid_y = y_num.notna()
    if int(valid_y.sum()) < 20 or int(y_num.loc[valid_y].nunique()) < 2:
        return _frame([], cols), ()
    rows: list[dict[str, object]] = []
    flagged: list[str] = []
    for col in df.columns:
        if col == tc:
            continue
        s = _s(df[col])
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            metric, score = _score_numeric_feature(s, y_num, valid_y)
        else:
            metric, score = _score_categorical_feature(s, y_num, valid_y)
        is_flagged = float(score) >= float(threshold)
        if is_flagged:
            flagged.append(str(col))
        rows.append(
            {
                "feature": str(col),
                "metric": metric,
                "score": round(float(score), 4),
                "flagged": is_flagged,
            }
        )
    out = _frame(rows, cols)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, tuple(flagged)


def collect_synthetic_notes(
    qa: QualityAuditResult,
    shape: pd.DataFrame,
    flagged: tuple[str, ...],
    combos: CombinationResult,
) -> tuple[str, ...]:
    notes: list[str] = []
    if qa.duplicate_count == 0:
        notes.append("Zero exact duplicate rows detected.")
    if not shape.empty and "outlier_rate_pct" in shape.columns:
        max_out = float(
            cast(Any, pd.to_numeric(shape["outlier_rate_pct"], errors="coerce")).max()
        )
        if np.isfinite(max_out) and max_out < 1.0:
            notes.append(
                f"Low numeric outlier rate (max {max_out:.2f}%), consistent with clean/synthetic bounds."
            )
    if flagged:
        notes.append(
            f"Leakage check flagged {len(flagged)} column(s): {', '.join(flagged)}."
        )
    if combos.total_combos > 0 and combos.replication_rate >= 0.8:
        notes.append(
            f"High split-half combination replication ({combos.replication_rate:.1%})."
        )
    return tuple(notes)
