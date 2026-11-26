from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from src.common.dataset.columns import (
    UNIDENTIFIED,
    DatasetSchema,
    is_identified,
    resolve_drivers,
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
