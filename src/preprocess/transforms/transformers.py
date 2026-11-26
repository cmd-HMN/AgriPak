from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.core.validation import (
    UNIDENTIFIED,
    as_series,
    require_non_empty,
    to_numeric_series,
)
from src.common.dataset.columns import (
    DatasetSchema,
    extract_binary_target,
    infer_schema,
    resolve_drivers,
    series_to_float,
)


@dataclass(frozen=True)
class EngineeredFeaturesResult:
    df: pd.DataFrame
    engineered_cols: list[str]
    binary_signal_cols: list[str]
    primary_driver: str
    risk_floor_threshold: float
    signal_sum_col: str
    deficit_col: str
    deficit_x_zero_col: str
    driver_x_sum_col: str


def _coerce_binary_series(series: pd.Series) -> pd.Series:
    return series_to_float(series).fillna(0.0)


def infer_risk_floor_threshold(
    df: pd.DataFrame, primary_driver: str, target_col: str | None
) -> float:
    if primary_driver == UNIDENTIFIED or primary_driver not in df.columns:
        return 0.0
    driver_series = to_numeric_series(as_series(df, primary_driver))
    driver_vals = driver_series.dropna()
    if driver_vals.empty:
        return 0.0
    if target_col and target_col in df.columns:
        target = extract_binary_target(df, target_col)
        pos = driver_series.loc[target == 1].dropna()
        if not pos.empty:
            return float(pos.median())
    return float(driver_vals.median())


def add_binary_signal_sum(
    df: pd.DataFrame, binary_cols: list[str], col_name: str = "binary_signal_sum"
) -> pd.DataFrame:
    out = df.copy()
    if not binary_cols:
        out[col_name] = 0.0
        return out
    matrix = pd.DataFrame(
        {
            c: _coerce_binary_series(as_series(out, c))
            for c in binary_cols
            if c in out.columns
        },
        index=out.index,
    )
    out[col_name] = matrix.sum(axis=1).astype(float) if not matrix.empty else 0.0
    return out


def add_driver_deficit(
    df: pd.DataFrame,
    primary_driver: str,
    threshold: float,
    col_name: str = "driver_deficit",
) -> pd.DataFrame:
    out = df.copy()
    if primary_driver == UNIDENTIFIED or primary_driver not in out.columns:
        out[col_name] = 0.0
        return out
    vals = to_numeric_series(as_series(out, primary_driver)).fillna(threshold)
    out[col_name] = np.maximum(0.0, threshold - vals).astype(float)
    return out


def add_deficit_x_zero_signal(
    df: pd.DataFrame,
    deficit_col: str = "driver_deficit",
    signal_sum_col: str = "binary_signal_sum",
    col_name: str = "deficit_x_zero_signal",
) -> pd.DataFrame:
    out = df.copy()
    deficit = (
        to_numeric_series(as_series(out, deficit_col)).fillna(0.0)
        if deficit_col in out.columns
        else pd.Series(0.0, index=out.index)
    )
    sig_sum = (
        to_numeric_series(as_series(out, signal_sum_col)).fillna(0.0)
        if signal_sum_col in out.columns
        else pd.Series(0.0, index=out.index)
    )
    zero_flag = (sig_sum == 0.0).astype(float)
    out[col_name] = (deficit * zero_flag).astype(float)
    return out


def add_driver_x_signal_sum(
    df: pd.DataFrame,
    primary_driver: str,
    signal_sum_col: str = "binary_signal_sum",
    col_name: str = "driver_x_signal_sum",
) -> pd.DataFrame:
    out = df.copy()
    if primary_driver == UNIDENTIFIED or primary_driver not in out.columns:
        out[col_name] = 0.0
        return out
    driver = to_numeric_series(as_series(out, primary_driver)).fillna(0.0)
    sig_sum = (
        to_numeric_series(as_series(out, signal_sum_col)).fillna(0.0)
        if signal_sum_col in out.columns
        else pd.Series(0.0, index=out.index)
    )
    out[col_name] = (driver * sig_sum).astype(float)
    return out


def engineer_features(
    df: pd.DataFrame, schema: DatasetSchema | None = None
) -> EngineeredFeaturesResult:
    require_non_empty(df, "df")
    active_schema = resolve_drivers(df, schema or infer_schema(df))
    primary_driver = active_schema.primary_driver
    binary_cols = [c for c in active_schema.binary_columns if c in df.columns]
    threshold = infer_risk_floor_threshold(
        df, primary_driver, active_schema.target_column
    )
    sig_col = "binary_signal_sum"
    def_col = "driver_deficit"
    dz_col = "deficit_x_zero_signal"
    ds_col = "driver_x_signal_sum"
    work = add_binary_signal_sum(df, binary_cols, sig_col)
    work = add_driver_deficit(work, primary_driver, threshold, def_col)
    work = add_deficit_x_zero_signal(work, def_col, sig_col, dz_col)
    work = add_driver_x_signal_sum(work, primary_driver, sig_col, ds_col)
    engineered = [sig_col, def_col, dz_col, ds_col]
    return EngineeredFeaturesResult(
        df=work,
        engineered_cols=engineered,
        binary_signal_cols=binary_cols,
        primary_driver=primary_driver,
        risk_floor_threshold=threshold,
        signal_sum_col=sig_col,
        deficit_col=def_col,
        deficit_x_zero_col=dz_col,
        driver_x_sum_col=ds_col,
    )
