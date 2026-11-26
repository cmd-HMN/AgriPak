from dataclasses import dataclass, replace
from typing import Callable, cast

import pandas as pd

from src.common.core.validation import (
    DEFAULT_BINARY_MAP,
    DEFAULT_MISSING_TOKENS,
    UNIDENTIFIED,
    as_series,
    is_identified,
    to_float,
    to_numeric_series,
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


def apply_to_columns(
    frame: pd.DataFrame,
    columns: list[str],
    transform: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = transform(as_series(out, col))
    return out


def is_text_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def series_to_text(
    series: pd.Series,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> pd.Series:
    text = series.astype("string").str.strip()
    lower = text.str.lower()
    return text.mask(text.isna() | (text == "") | lower.isin(missing_tokens))


def series_to_float(
    series: pd.Series,
    binary_map: dict[str, int] = DEFAULT_BINARY_MAP,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return to_numeric_series(series).astype(float)
    cleaned = series_to_text(series).str.lower()
    mapped = cleaned.map(binary_map)
    if int(mapped.notna().sum()) >= max(1, int(cleaned.notna().sum() * 0.8)):
        return to_numeric_series(mapped).astype(float)
    return to_numeric_series(cleaned).astype(float)


def _classify_numeric(vals: pd.Series, max_cat: int, is_int: bool = False) -> str:
    nunique = int(vals.nunique(dropna=True))
    if nunique <= 1:
        return "unidentified"
    if nunique == 2:
        return "binary"
    if nunique <= max_cat and (is_int or bool((vals % 1 == 0).all())):
        return "numeric_discrete"
    return "numeric_continuous"


def _classify_non_numeric(
    series: pd.Series,
    max_cat: int,
    missing_tokens: frozenset[str],
) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    cleaned = series_to_text(series, missing_tokens)
    non_null = cleaned.dropna()
    if non_null.empty:
        return "unidentified"
    lowered = non_null.str.lower()
    if lowered.isin(DEFAULT_BINARY_MAP).all() and int(lowered.nunique()) <= 2:
        return "binary"
    num_vals = to_numeric_series(non_null)
    if float(num_vals.notna().mean()) >= 0.9:
        return _classify_numeric(num_vals.dropna(), max_cat)
    if float(non_null.str.len().mean()) > 48.0 and int(non_null.nunique()) > max_cat:
        return "text"
    if int(non_null.nunique()) <= max(max_cat * 4, 80):
        return "categorical"
    return "text"


def _classify_column(
    series: pd.Series,
    max_cat: int,
    missing_tokens: frozenset[str],
) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "binary" if int(series.dropna().nunique()) == 2 else "unidentified"
    if pd.api.types.is_numeric_dtype(series):
        vals = to_numeric_series(series).dropna()
        if vals.empty:
            return "unidentified"
        return _classify_numeric(vals, max_cat, pd.api.types.is_integer_dtype(series))
    return _classify_non_numeric(series, max_cat, missing_tokens)


def _abs_spearman(s1: pd.Series, s2: pd.Series) -> float | None:
    pair = pd.DataFrame({"a": s1, "b": s2}).dropna()
    sa = as_series(pair, "a")
    sb = as_series(pair, "b")
    if len(pair) < 3 or int(sa.nunique()) < 2 or int(sb.nunique()) < 2:
        return None
    corr = float(sa.corr(sb, method="spearman"))
    return None if pd.isna(corr) else abs(corr)


def _select_target(
    df: pd.DataFrame,
    kinds: dict[str, str],
    target_column: str | None,
) -> tuple[str, str]:
    if target_column is not None and target_column in df.columns:
        return target_column, kinds.get(target_column, "unidentified")
    for kind in ("binary", "numeric_continuous", "numeric_discrete", "categorical"):
        cands = [col for col, k in kinds.items() if k == kind]
        if not cands:
            continue
        if kind == "binary":
            return max(
                cands,
                key=lambda c: to_float(series_to_float(as_series(df, c)).std(), 0.0),
            ), kind
        if kind in {"numeric_continuous", "numeric_discrete"}:
            return max(
                cands,
                key=lambda c: to_float(to_numeric_series(as_series(df, c)).var(), 0.0),
            ), kind
        return max(
            cands, key=lambda c: int(as_series(df, c).nunique(dropna=True))
        ), kind
    return UNIDENTIFIED, "unidentified"


def _select_subgroup(
    df: pd.DataFrame,
    kinds: dict[str, str],
    target: str,
) -> str:
    cands = [
        c for c, k in kinds.items() if k in {"categorical", "binary"} and c != target
    ]
    if not cands:
        return UNIDENTIFIED
    preferred = [
        c for c in cands if 2 <= int(as_series(df, c).nunique(dropna=True)) <= 20
    ]
    pool = preferred if preferred else cands
    return max(pool, key=lambda c: int(as_series(df, c).nunique(dropna=True)))


def resolve_drivers(df: pd.DataFrame, schema: DatasetSchema) -> DatasetSchema:
    target = schema.target_column
    candidates = [
        c for c in schema.numeric_columns + schema.binary_flags if c != target
    ]
    if not candidates:
        return schema
    scored: list[tuple[float, str]] = []
    if is_identified(target, df):
        t_series = series_to_float(as_series(df, target))
        for col in candidates:
            corr = _abs_spearman(series_to_float(as_series(df, col)), t_series)
            if corr is not None:
                scored.append((corr, col))
    if not scored:
        scored = [
            (to_float(series_to_float(as_series(df, c)).var(), 0.0), c)
            for c in candidates
        ]
    ordered = [col for _, col in sorted(scored, key=lambda item: item[0], reverse=True)]
    primary = ordered[0] if len(ordered) >= 1 else UNIDENTIFIED
    secondary = ordered[1] if len(ordered) >= 2 else UNIDENTIFIED
    return replace(schema, primary_driver=primary, secondary_driver=secondary)


def get_drivers(schema: DatasetSchema, df: pd.DataFrame | None = None) -> list[str]:
    drivers = [schema.primary_driver, schema.secondary_driver]
    return [col for col in drivers if is_identified(col, df)]


def extract_target_series(df: pd.DataFrame, target_column: str) -> pd.Series:
    return series_to_float(as_series(df, target_column))


def extract_binary_target(
    df: pd.DataFrame,
    target_column: str,
    binary_map: dict[str, int] = DEFAULT_BINARY_MAP,
) -> pd.Series:
    return series_to_float(as_series(df, target_column), binary_map=binary_map)


def _build_profiles(
    df: pd.DataFrame,
    kinds: dict[str, str],
    target: str,
    subgroup: str,
) -> tuple[ColumnProfile, ...]:
    n_rows = max(1, len(df))
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = as_series(df, str(col))
        null_cnt = int(s.isna().sum())
        role = (
            "target"
            if col == target
            else ("subgroup" if col == subgroup else "feature")
        )
        profiles.append(
            ColumnProfile(
                name=str(col),
                kind=kinds[str(col)],
                role=role,
                dtype=str(s.dtype),
                nunique=int(s.nunique(dropna=True)),
                null_count=null_cnt,
                null_pct=float(null_cnt / n_rows),
            )
        )
    return tuple(profiles)


def infer_schema(
    df: pd.DataFrame,
    target_column: str | None = None,
    max_categorical_nunique: int = 20,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> DatasetSchema:
    kinds = {
        str(col): _classify_column(
            as_series(df, str(col)), max_categorical_nunique, missing_tokens
        )
        for col in df.columns
    }
    target, target_kind = _select_target(df, kinds, target_column)
    subgroup = _select_subgroup(df, kinds, target)
    schema = DatasetSchema(
        numeric_continuous=tuple(
            c for c, k in kinds.items() if k == "numeric_continuous"
        ),
        numeric_discrete=tuple(c for c, k in kinds.items() if k == "numeric_discrete"),
        binary_flags=tuple(c for c, k in kinds.items() if k == "binary"),
        categorical=tuple(c for c, k in kinds.items() if k == "categorical"),
        datetime_columns=tuple(c for c, k in kinds.items() if k == "datetime"),
        text_columns=tuple(c for c, k in kinds.items() if k == "text"),
        unidentified_columns=tuple(c for c, k in kinds.items() if k == "unidentified"),
        target_column=target,
        target_kind=target_kind,
        primary_driver=UNIDENTIFIED,
        secondary_driver=UNIDENTIFIED,
        primary_subgroup=subgroup,
        profiles=cast(
            tuple[ColumnProfile, ...], _build_profiles(df, kinds, target, subgroup)
        ),
    )
    return resolve_drivers(df, schema)
