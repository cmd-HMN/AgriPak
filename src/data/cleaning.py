from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.errors import EmptyDataError
from src.data.schema import DatasetSchema, infer_schema, resolve_drivers
from src.data.utils import (
    DEFAULT_BINARY_MAP,
    DEFAULT_MISSING_TOKENS,
    UNIDENTIFIED,
    as_series,
    dataframe_to_markdown,
    is_identified,
    normalize_label,
    to_float,
    to_int,
    to_numeric_series,
)


@dataclass
class CleaningEntry:
    step: str
    issue: str
    action: str
    rows_affected: int


@dataclass
class CleaningAudit:
    entries: list[CleaningEntry] = field(default_factory=list)

    def add(self, step: str, issue: str, action: str, rows_affected: int) -> None:
        self.entries.append(CleaningEntry(step, issue, action, int(rows_affected)))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["step", "issue", "action", "rows_affected"])
        return pd.DataFrame(
            [
                {
                    "step": e.step,
                    "issue": e.issue,
                    "action": e.action,
                    "rows_affected": e.rows_affected,
                }
                for e in self.entries
            ]
        )

    def to_markdown(self) -> str:
        return dataframe_to_markdown(self.to_dataframe())


@dataclass
class CleaningResult:
    data: pd.DataFrame
    audit: CleaningAudit
    schema: DatasetSchema
    rows_before: int
    rows_after: int
    memory_mb_before: float
    memory_mb_after: float


def _memory_usage_mb(df: pd.DataFrame) -> float:
    return round(to_float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0), 4)


def _compact_int_dtype(series: pd.Series) -> str:
    if bool(series.isna().any()):
        return "Int64"
    lo, hi = to_int(series.min()), to_int(series.max())
    for name, np_type in (("int8", np.int8), ("int16", np.int16), ("int32", np.int32)):
        info = np.iinfo(np_type)
        if lo >= info.min and hi <= info.max:
            return name
    return "int64"


def normalize_text_columns(
    df: pd.DataFrame,
    schema: DatasetSchema,
    audit: CleaningAudit,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> pd.DataFrame:
    result = df.copy()
    cols = list(
        schema.categorical + schema.text_columns + schema.unidentified_columns
    ) + [
        c
        for c in schema.binary_flags
        if c in result.columns
        and not pd.api.types.is_numeric_dtype(as_series(result, c).dtype)
    ]
    for col in cols:
        if col not in result.columns:
            continue
        orig = as_series(result, col).astype("object")
        norm = orig.map(
            lambda v: (
                np.nan
                if (lbl := normalize_label(v, missing_tokens)) == UNIDENTIFIED
                else lbl
            )
        )
        affected = to_int(
            (
                ~((orig.isna() & norm.isna()) | (orig.astype(str) == norm.astype(str)))
            ).sum()
        )
        result[col] = norm
        if affected > 0:
            audit.add(
                "normalize_text",
                f"Whitespace or placeholder tokens in '{col}'",
                "Stripped whitespace and standardized missing tokens",
                affected,
            )
    return result


def coerce_column_types(
    df: pd.DataFrame,
    schema: DatasetSchema,
    audit: CleaningAudit,
    binary_map: Mapping[str, int] = DEFAULT_BINARY_MAP,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> pd.DataFrame:
    result = df.copy()
    for cols, converter, issue_tpl, action in (
        (
            schema.numeric_columns,
            to_numeric_series,
            "Non-numeric values in '{}'",
            "Coerced to numeric with invalid values set to NaN",
        ),
        (
            schema.datetime_columns,
            lambda s: pd.to_datetime(s, errors="coerce"),
            "Unparseable datetime values in '{}'",
            "Coerced to datetime with invalid values set to NaT",
        ),
    ):
        for col in cols:
            if col not in result.columns:
                continue
            series = as_series(result, col)
            coerced = converter(series)
            newly_na = max(
                0, to_int(coerced.isna().sum()) - to_int(series.isna().sum())
            )
            result[col] = coerced
            if newly_na > 0:
                audit.add("coerce_types", issue_tpl.format(col), action, newly_na)

    for col in schema.binary_flags:
        if col not in result.columns:
            continue
        series = as_series(result, col)
        if pd.api.types.is_bool_dtype(series.dtype):
            coerced = series.astype("float64")
        elif pd.api.types.is_numeric_dtype(series.dtype):
            coerced = to_numeric_series(series).astype("float64")
        else:
            tokens = series.astype(str).str.strip().str.lower()
            coerced = (
                tokens.where(
                    ~series.isna() & ~tokens.isin(missing_tokens), other=np.nan
                )
                .map(binary_map)
                .astype("float64")
            )
        newly_na = max(0, to_int(coerced.isna().sum()) - to_int(series.isna().sum()))
        result[col] = coerced
        if newly_na > 0 or not pd.api.types.is_numeric_dtype(series.dtype):
            audit.add(
                "coerce_types",
                f"Unstandardized binary values in '{col}'",
                "Mapped binary values to 0/1 numeric representation",
                max(newly_na, to_int(series.notna().sum())),
            )
    return result


def validate_numeric_ranges(
    df: pd.DataFrame, schema: DatasetSchema, audit: CleaningAudit
) -> pd.DataFrame:
    result = df.copy()
    for col in schema.numeric_columns:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(
            as_series(result, col).dtype
        ):
            continue
        series = as_series(result, col)
        if (
            inf_count := to_int(
                np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()
            )
        ) > 0:
            result[col] = series.replace([np.inf, -np.inf], np.nan)
            audit.add(
                "validate_numeric_ranges",
                f"Infinite values detected in '{col}'",
                "Replaced +/-inf with NaN",
                inf_count,
            )
    return result


def handle_missing_and_duplicates(
    df: pd.DataFrame,
    schema: DatasetSchema,
    audit: CleaningAudit,
    fallback_label: str = UNIDENTIFIED,
) -> pd.DataFrame:
    result = df.copy()
    if is_identified(schema.target_column, result):
        missing_target = as_series(result, schema.target_column).isna()
        if (dropped := to_int(missing_target.sum())) > 0:
            result = result.loc[~missing_target].copy()
            audit.add(
                "handle_missing",
                f"Missing target values in '{schema.target_column}'",
                "Dropped rows with missing target",
                dropped,
            )
        if result.empty:
            raise EmptyDataError(
                f"No rows remain after dropping missing target '{schema.target_column}'"
            )

    for col in schema.categorical + schema.text_columns + schema.unidentified_columns:
        if col not in result.columns or col == schema.target_column:
            continue
        series = as_series(result, col)
        if (n_miss := to_int(series.isna().sum())) > 0:
            if not pd.api.types.is_string_dtype(
                series.dtype
            ) and not pd.api.types.is_object_dtype(series.dtype):
                series = series.astype("object")
            result[col] = series.fillna(fallback_label)
            audit.add(
                "handle_missing",
                f"Missing categorical or text values in '{col}'",
                f"Filled missing values with '{fallback_label}'",
                n_miss,
            )

    for col in schema.numeric_columns:
        if col not in result.columns or col == schema.target_column:
            continue
        series = as_series(result, col)
        if (n_miss := to_int(series.isna().sum())) > 0:
            non_null = series.dropna()
            fill_val = to_float(non_null.median()) if not non_null.empty else 0.0
            result[col] = series.fillna(fill_val)
            audit.add(
                "handle_missing",
                f"Missing numeric values in '{col}'",
                f"Imputed missing values with median ({fill_val:.4g})",
                n_miss,
            )

    for col in schema.binary_flags:
        if col not in result.columns or col == schema.target_column:
            continue
        series = as_series(result, col)
        if (n_miss := to_int(series.isna().sum())) > 0:
            non_null = series.dropna()
            fill_val = (
                int(round(to_float(non_null.mode().iloc[0])))
                if not non_null.empty
                else 0
            )
            result[col] = series.fillna(fill_val)
            audit.add(
                "handle_missing",
                f"Missing binary flag values in '{col}'",
                f"Imputed missing values with mode ({fill_val})",
                n_miss,
            )

    before_dedup = len(result)
    result = result.drop_duplicates().copy()
    if (dedup_dropped := before_dedup - len(result)) > 0:
        audit.add(
            "deduplicate",
            "Exact duplicate rows detected",
            "Dropped duplicate rows",
            dedup_dropped,
        )
    if result.empty:
        raise EmptyDataError("No rows remain after deduplication")
    return result


def finalize_dtypes(
    df: pd.DataFrame, schema: DatasetSchema, audit: CleaningAudit
) -> pd.DataFrame:
    result = df.copy()
    cast_count = 0
    for col in schema.categorical:
        if col in result.columns:
            result[col] = as_series(result, col).astype("category")
            cast_count += 1
    for col in schema.numeric_continuous:
        if col in result.columns:
            result[col] = to_numeric_series(as_series(result, col)).astype("float64")
            cast_count += 1
    for col in schema.numeric_discrete:
        if col in result.columns:
            num = to_numeric_series(as_series(result, col)).round()
            result[col] = num.astype(
                "Int64" if bool(num.isna().any()) else _compact_int_dtype(num)
            )
            cast_count += 1
    for col in schema.binary_flags:
        if col in result.columns:
            num = to_numeric_series(as_series(result, col)).round()
            result[col] = num.astype("Int8" if bool(num.isna().any()) else "int8")
            cast_count += 1
    result = result.reset_index(drop=True)
    if cast_count > 0:
        audit.add(
            "finalize_dtypes",
            "Unoptimized or mixed column dtypes",
            f"Finalized dtypes across {cast_count} columns and reset index",
            len(result),
        )
    return result


def clean_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
    binary_map: Mapping[str, int] = DEFAULT_BINARY_MAP,
    fallback_label: str = UNIDENTIFIED,
    on_step: Callable[[str], None] | None = None,
) -> CleaningResult:
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError("Cannot clean an empty DataFrame")

    def _tick(label: str) -> None:
        if on_step is not None:
            on_step(label)

    rows_before, memory_mb_before = len(df), _memory_usage_mb(df)
    initial = infer_schema(
        df,
        target_column=target_column,
        missing_tokens=missing_tokens,
        binary_map=binary_map,
    )
    audit = CleaningAudit()
    _tick("Normalizing text columns...")
    cleaned = normalize_text_columns(df, initial, audit, missing_tokens=missing_tokens)
    _tick("Coercing column types...")
    cleaned = coerce_column_types(
        cleaned, initial, audit, binary_map=binary_map, missing_tokens=missing_tokens
    )
    _tick("Validating numeric ranges...")
    cleaned = validate_numeric_ranges(cleaned, initial, audit)
    _tick("Handling missing values and duplicates...")
    cleaned = handle_missing_and_duplicates(
        cleaned, initial, audit, fallback_label=fallback_label
    )
    _tick("Finalizing dtypes...")
    cleaned = finalize_dtypes(cleaned, initial, audit)
    return CleaningResult(
        cleaned,
        audit,
        resolve_drivers(cleaned, initial),
        rows_before,
        len(cleaned),
        memory_mb_before,
        _memory_usage_mb(cleaned),
    )
