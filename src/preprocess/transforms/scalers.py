from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.common.core.errors import TransformError
from src.common.core.validation import as_series, require_non_empty, to_numeric_series
from src.common.dataset.columns import apply_to_columns


@dataclass(frozen=True)
class ScalingResult:
    df: pd.DataFrame
    scaled_columns: list[str]
    zero_variance_columns: list[str]
    method: str
    means: dict[str, float]
    stds: dict[str, float]


def _build_standard() -> StandardScaler:
    return StandardScaler()


def _build_robust() -> RobustScaler:
    return RobustScaler()


def _build_minmax() -> MinMaxScaler:
    return MinMaxScaler()


_SCALERS: dict[str, Callable[[], Any]] = {
    "standard": _build_standard,
    "robust": _build_robust,
    "minmax": _build_minmax,
}


def build_numeric_scaler(method: str = "standard") -> Any:
    key = method.strip().lower()
    if key not in _SCALERS:
        raise TransformError(
            f"Unsupported scaling method: {method}. Available: {sorted(_SCALERS)}"
        )
    return _SCALERS[key]()


def _scale_series(series: pd.Series, method: str) -> pd.Series:
    numeric = to_numeric_series(series)
    valid = numeric.dropna().to_numpy(dtype=float)
    if valid.size == 0 or float(np.std(valid)) == 0.0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    filled = numeric.fillna(float(np.median(valid)))
    scaler = build_numeric_scaler(method)
    scaled = scaler.fit_transform(filled.to_numpy(dtype=float).reshape(-1, 1)).ravel()
    return pd.Series(np.nan_to_num(scaled, nan=0.0), index=series.index)


def scale_numeric_columns(
    df: pd.DataFrame, columns: list[str], method: str = "standard"
) -> ScalingResult:
    require_non_empty(df, "df")
    build_numeric_scaler(method)
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return ScalingResult(
            df=df.copy(),
            scaled_columns=[],
            zero_variance_columns=[],
            method=method.strip().lower(),
            means={},
            stds={},
        )
    zero_var: list[str] = []
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in cols:
        valid = to_numeric_series(as_series(df, col)).dropna().to_numpy(dtype=float)
        std_val = float(np.std(valid)) if valid.size > 0 else 0.0
        means[col] = float(np.mean(valid)) if valid.size > 0 else 0.0
        stds[col] = std_val
        if std_val == 0.0:
            zero_var.append(col)
    scaled_df = apply_to_columns(df, cols, lambda s: _scale_series(s, method))
    return ScalingResult(
        df=scaled_df,
        scaled_columns=cols,
        zero_variance_columns=zero_var,
        method=method.strip().lower(),
        means=means,
        stds=stds,
    )
