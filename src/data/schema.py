from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.errors import EmptyDataError, MissingColumnError
from src.data.utils import (
    DEFAULT_BINARY_MAP,
    DEFAULT_MISSING_TOKENS,
    UNIDENTIFIED,
    as_series,
    is_identified,
    to_float,
    to_int,
    to_numeric_series,
)

_KINDS = (
    "numeric_continuous",
    "numeric_discrete",
    "binary_flag",
    "categorical",
    "datetime",
    "text",
    "unidentified",
)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    kind: str
    role: str
    dtype: str
    nunique: int
    null_count: int
    null_pct: float


@dataclass(frozen=True)
class DatasetSchema:
    numeric_continuous: tuple[str, ...]
    numeric_discrete: tuple[str, ...]
    binary_flags: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    text_columns: tuple[str, ...]
    unidentified_columns: tuple[str, ...]
    target_column: str
    target_kind: str
    primary_driver: str
    secondary_driver: str
    primary_subgroup: str
    profiles: tuple[ColumnProfile, ...]

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return self.numeric_continuous + self.numeric_discrete

    @property
    def binary_columns(self) -> tuple[str, ...]:
        return self.binary_flags

    @property
    def categorical_and_flags(self) -> tuple[str, ...]:
        return self.categorical + self.binary_flags

    @property
    def all_feature_columns(self) -> tuple[str, ...]:
        ordered = (
            self.numeric_continuous
            + self.numeric_discrete
            + self.binary_flags
            + self.categorical
            + self.datetime_columns
            + self.text_columns
        )
        return tuple(col for col in ordered if col != self.target_column)


def _clean_text(series: pd.Series, missing_tokens: frozenset[str]) -> pd.Series:
    non_null = (
        series.dropna().astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    return non_null[~non_null.str.lower().isin(missing_tokens)]


def _classify_numeric(vals: pd.Series, max_cat: int, is_int: bool = False) -> str:
    clean = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return "unidentified"
    uniq = set(clean.unique().tolist())
    if len(uniq) == 2 and uniq.issubset({0, 1, 0.0, 1.0}):
        return "binary_flag"
    if is_int:
        return "numeric_discrete"
    arr = clean.to_numpy(dtype=float)
    if np.all(np.isclose(arr, np.round(arr))) and len(uniq) <= max_cat:
        return "numeric_discrete"
    return "numeric_continuous"


def _classify_column(
    series: pd.Series,
    missing_tokens: frozenset[str],
    binary_map: Mapping[str, int],
    max_cat: int,
) -> tuple[str, int, int, float]:
    total = len(series)
    if total == 0:
        return "unidentified", 0, 0, 0.0
    dtype = series.dtype
    is_text = (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )
    if not is_text:
        null_count = to_int(series.isna().sum())
        non_null = series.dropna()
        null_pct = round((null_count / total) * 100.0, 2)
        if non_null.empty:
            return "unidentified", 0, null_count, null_pct
        if pd.api.types.is_bool_dtype(dtype):
            kind = "binary_flag"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(dtype):
            kind = _classify_numeric(
                non_null, max_cat, bool(pd.api.types.is_integer_dtype(dtype))
            )
        else:
            kind = "unidentified"
        return kind, to_int(non_null.nunique()), null_count, null_pct

    valid = _clean_text(series, missing_tokens)
    null_count = total - len(valid)
    null_pct = round((null_count / total) * 100.0, 2)
    if valid.empty:
        return "unidentified", 0, null_count, null_pct

    lower_uniq = set(valid.str.lower().unique().tolist())
    if lower_uniq.issubset(binary_map.keys()):
        mapped = {binary_map[tok] for tok in lower_uniq}
        if len(mapped) <= 2:
            return "binary_flag", len(mapped), null_count, null_pct

    parsed = to_numeric_series(valid)
    if to_int(parsed.isna().sum()) == 0:
        return (
            _classify_numeric(parsed, max_cat),
            to_int(parsed.nunique()),
            null_count,
            null_pct,
        )

    nunique = to_int(valid.nunique())
    kind = (
        "categorical"
        if isinstance(dtype, pd.CategoricalDtype) or nunique <= max_cat
        else "text"
    )
    return kind, nunique, null_count, null_pct


def _series_to_float(
    series: pd.Series,
    binary_map: Mapping[str, int] = DEFAULT_BINARY_MAP,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return to_numeric_series(series).astype("float64")
    text = series.astype(str).str.strip().str.lower()
    text = text.where(~series.isna() & ~text.isin(missing_tokens), other=np.nan)
    mapped = text.map(binary_map)
    n_valid = to_int(text.notna().sum())
    if n_valid > 0 and to_int(mapped.notna().sum()) == n_valid:
        return mapped.astype("float64")
    return to_numeric_series(text).astype("float64")


def _abs_spearman(s1: pd.Series, s2: pd.Series) -> float | None:
    paired = (
        pd.DataFrame({"a": s1, "b": s2}).replace([np.inf, -np.inf], np.nan).dropna()
    )
    if (
        len(paired) < 3
        or to_int(paired["a"].nunique()) < 2
        or to_int(paired["b"].nunique()) < 2
    ):
        return None
    res: Any = spearmanr(paired["a"].to_numpy(), paired["b"].to_numpy())
    stat = to_float(getattr(res, "statistic", float("nan")), float("nan"))
    return None if math.isnan(stat) else abs(stat)


def _select_target(
    df: pd.DataFrame,
    binary_flags: tuple[str, ...],
    num_cont: tuple[str, ...],
    num_disc: tuple[str, ...],
    binary_map: Mapping[str, int],
    missing_tokens: frozenset[str],
) -> str:
    if not binary_flags:
        return (
            num_cont[-1] if num_cont else (num_disc[-1] if num_disc else UNIDENTIFIED)
        )
    cols, peers = list(df.columns), num_cont + num_disc
    cache = {
        c: _series_to_float(as_series(df, c), binary_map, missing_tokens) for c in peers
    }
    best_col, best_score, denom = binary_flags[0], -1.0, max(1, len(cols) - 1)
    for col in binary_flags:
        num = _series_to_float(as_series(df, col), binary_map, missing_tokens)
        clean = num.dropna()
        balance = (
            0.0
            if to_int(clean.nunique()) < 2
            else 1.0
            - abs(
                2.0 * to_float((clean == sorted(clean.unique().tolist())[-1]).mean())
                - 1.0
            )
        )
        corrs = [
            r
            for p in peers
            if p != col and (r := _abs_spearman(num, cache[p])) is not None
        ]
        score = (
            0.4 * (to_float(np.mean(corrs)) if corrs else 0.0)
            + 0.35 * balance
            + 0.25 * (cols.index(col) / denom)
        )
        if score > best_score:
            best_score, best_col = score, col
    return best_col


def _select_subgroup(
    df: pd.DataFrame,
    categorical: tuple[str, ...],
    nunique_map: Mapping[str, int],
    missing_tokens: frozenset[str],
) -> str:
    if not categorical:
        return UNIDENTIFIED
    preferred = [c for c in categorical if 3 <= nunique_map.get(c, 0) <= 12]
    if not preferred:
        return categorical[0]

    def _entropy(col: str) -> float:
        probs = (
            _clean_text(as_series(df, col), missing_tokens)
            .value_counts(normalize=True)
            .to_numpy(dtype=float)
        )
        return (
            0.0
            if len(probs) <= 1
            else to_float(-np.sum(probs * np.log2(probs))) / math.log2(len(probs))
        )

    return max(preferred, key=_entropy)


def resolve_drivers(df: pd.DataFrame, schema: DatasetSchema) -> DatasetSchema:
    primary, secondary = UNIDENTIFIED, UNIDENTIFIED
    if is_identified(schema.target_column, df):
        target_s = _series_to_float(as_series(df, schema.target_column))
        scored = sorted(
            (
                (corr, -idx, col)
                for idx, col in enumerate(schema.numeric_columns)
                if col != schema.target_column
                and col in df.columns
                and (
                    corr := _abs_spearman(
                        _series_to_float(as_series(df, col)), target_s
                    )
                )
                is not None
            ),
            reverse=True,
        )
        if len(scored) >= 1:
            primary = scored[0][2]
        if len(scored) >= 2:
            secondary = scored[1][2]
    role_map: dict[str, str] = {}
    for col, role in (
        (schema.target_column, "target"),
        (primary, "primary_driver"),
        (secondary, "secondary_driver"),
        (schema.primary_subgroup, "primary_subgroup"),
    ):
        if col != UNIDENTIFIED and col not in role_map:
            role_map[col] = role
    profiles = tuple(
        replace(p, role=role_map.get(p.name, "feature")) for p in schema.profiles
    )
    return replace(
        schema, primary_driver=primary, secondary_driver=secondary, profiles=profiles
    )


def infer_schema(
    df: pd.DataFrame,
    target_column: str | None = None,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
    binary_map: Mapping[str, int] = DEFAULT_BINARY_MAP,
    max_categorical_cardinality: int = 30,
) -> DatasetSchema:
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError("Cannot infer schema from an empty DataFrame")
    if target_column is not None and target_column not in df.columns:
        raise MissingColumnError(
            f"Target column '{target_column}' not found in DataFrame"
        )

    buckets: dict[str, list[str]] = {k: [] for k in _KINDS}
    kind_map: dict[str, str] = {}
    nunique_map: dict[str, int] = {}
    raw_profiles: list[ColumnProfile] = []
    for col in df.columns:
        name = str(col)
        series = as_series(df, name)
        kind, nunique, null_count, null_pct = _classify_column(
            series, missing_tokens, binary_map, max_categorical_cardinality
        )
        buckets.setdefault(kind, buckets["unidentified"]).append(name)
        kind_map[name], nunique_map[name] = kind, nunique
        raw_profiles.append(
            ColumnProfile(
                name, kind, "feature", str(series.dtype), nunique, null_count, null_pct
            )
        )

    target = (
        target_column
        if target_column is not None
        else _select_target(
            df,
            tuple(buckets["binary_flag"]),
            tuple(buckets["numeric_continuous"]),
            tuple(buckets["numeric_discrete"]),
            binary_map,
            missing_tokens,
        )
    )
    initial = DatasetSchema(
        tuple(buckets["numeric_continuous"]),
        tuple(buckets["numeric_discrete"]),
        tuple(buckets["binary_flag"]),
        tuple(buckets["categorical"]),
        tuple(buckets["datetime"]),
        tuple(buckets["text"]),
        tuple(buckets["unidentified"]),
        target,
        kind_map.get(target, "unidentified"),
        UNIDENTIFIED,
        UNIDENTIFIED,
        _select_subgroup(
            df, tuple(buckets["categorical"]), nunique_map, missing_tokens
        ),
        tuple(raw_profiles),
    )
    return resolve_drivers(df, initial)
