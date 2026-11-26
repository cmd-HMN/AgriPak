import math

import numpy as np
import pandas as pd
from scipy import stats

from src.common.core.validation import as_series, to_float, to_numeric_series


def compute_cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype(str), right.astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    n = float(table.to_numpy().sum())
    if n <= 0.0:
        return 0.0
    chi2 = to_float(stats.chi2_contingency(table, correction=False)[0], 0.0)
    denom = n * float(min(table.shape[0] - 1, table.shape[1] - 1))
    if denom <= 0.0:
        return 0.0
    return float(math.sqrt(max(0.0, chi2 / denom)))


def _numeric_shape_row(col: str, vals: pd.Series) -> dict[str, object]:
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = int(((vals < lower) | (vals > upper)).sum())
    return {
        "column": col,
        "count": len(vals),
        "mean": float(vals.mean()),
        "std": to_float(vals.std(ddof=1), 0.0) if len(vals) > 1 else 0.0,
        "median": to_float(vals.median(), 0.0),
        "iqr": iqr,
        "skewness": to_float(vals.skew(), 0.0) if len(vals) > 2 else 0.0,
        "kurtosis": to_float(vals.kurt(), 0.0) if len(vals) > 3 else 0.0,
        "outlier_count": outliers,
        "outlier_pct": float(outliers / len(vals)) * 100.0,
    }


def compute_numeric_shape_metrics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = to_numeric_series(as_series(df, col)).dropna()
        if vals.empty:
            continue
        rows.append(_numeric_shape_row(col, vals))
    return pd.DataFrame(rows)


def compute_spearman_with_target(
    df: pd.DataFrame,
    cols: list[str],
    target_series: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in cols:
        if col not in df.columns:
            continue
        pair = pd.DataFrame(
            {
                "x": to_numeric_series(as_series(df, col)),
                "y": to_numeric_series(target_series),
            }
        ).dropna()
        sx = as_series(pair, "x")
        sy = as_series(pair, "y")
        if len(pair) < 3 or int(sx.nunique()) < 2 or int(sy.nunique()) < 2:
            continue
        res = stats.spearmanr(sx.to_numpy(), sy.to_numpy())
        rho = to_float(res[0], float("nan"))
        pval = to_float(res[1], 1.0)
        if np.isnan(rho):
            continue
        rows.append(
            {
                "column": col,
                "n": len(pair),
                "spearman_rho": rho,
                "abs_spearman": abs(rho),
                "p_value": pval,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["column", "n", "spearman_rho", "abs_spearman", "p_value"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_spearman", ascending=False)
        .reset_index(drop=True)
    )
