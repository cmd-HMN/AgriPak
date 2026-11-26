from collections.abc import Callable, Mapping

import pandas as pd

from src.common.core.errors import EmptyDataError
from src.common.core.validation import DEFAULT_BINARY_MAP, DEFAULT_MISSING_TOKENS
from src.common.dataset.columns import UNIDENTIFIED, infer_schema, resolve_drivers
from src.data.cleaning.imputation import finalize_dtypes, handle_missing_and_duplicates
from src.data.cleaning.sanitizers import (
    coerce_column_types,
    normalize_text_columns,
    validate_numeric_ranges,
)
from src.data.cleaning.types import CleaningAudit, CleaningResult, _memory_usage_mb


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
