import pandas as pd

from src.common.core.errors import EmptyDataError
from src.common.core.validation import as_series, to_float, to_int, to_numeric_series
from src.common.dataset.columns import UNIDENTIFIED, DatasetSchema, is_identified
from src.data.cleaning.types import CleaningAudit, _compact_int_dtype


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
