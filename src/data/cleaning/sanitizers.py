from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.common.core.validation import (
    DEFAULT_BINARY_MAP,
    DEFAULT_MISSING_TOKENS,
    as_series,
    normalize_label,
    to_int,
    to_numeric_series,
)
from src.common.dataset.columns import UNIDENTIFIED, DatasetSchema
from src.data.cleaning.types import CleaningAudit


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
