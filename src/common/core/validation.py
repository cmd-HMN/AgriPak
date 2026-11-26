import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.common.core.errors import (
    EmptyDataError,
    InvalidArgumentError,
    MissingColumnError,
    ReportWriteError,
)

UNIDENTIFIED = "Unidentified"

DEFAULT_MISSING_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "null", "none", "nan", "?", "-"}
)

DEFAULT_BINARY_MAP: dict[str, int] = {
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "y": 1,
    "n": 0,
}


def as_series(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, df[col])


def to_numeric_series(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def to_float(value: Any, default: float = 0.0) -> float:
    if is_missing_scalar(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if is_missing_scalar(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def normalize_label(
    value: object,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> str:
    if is_missing_scalar(value):
        return UNIDENTIFIED
    cleaned = " ".join(str(value).strip().split())
    if not cleaned or cleaned.lower() in missing_tokens:
        return UNIDENTIFIED
    return cleaned


def is_identified(name: str | None, df: pd.DataFrame | None = None) -> bool:
    if name is None:
        return False
    cleaned = str(name).strip()
    if not cleaned or cleaned == UNIDENTIFIED:
        return False
    return cleaned in df.columns if df is not None else True


def compute_min_subgroup_size(
    row_count: int,
    min_floor: int = 20,
    fraction: float = 0.01,
) -> int:
    floor_val = max(1, int(min_floor))
    if row_count <= 0:
        return floor_val
    scaled = int(math.ceil(float(row_count) * max(0.0, float(fraction))))
    return max(floor_val, scaled)


def require_non_empty(df: pd.DataFrame, context: str = "dataset") -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"{context} is empty.")
    return df


def require_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise MissingColumnError(
            f"Required columns missing from dataset: {', '.join(missing)}"
        )


def validate_min_int(value: int, min_value: int, name: str) -> int:
    if int(value) < int(min_value):
        raise InvalidArgumentError(f"{name} must be >= {min_value}, got {value}.")
    return int(value)


def validate_unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise InvalidArgumentError(f"{name} must be in [0.0, 1.0], got {value}.")
    return numeric


def ensure_not_overwriting_input(
    output_path: Path,
    input_path: Path | None,
) -> None:
    if input_path is None:
        return
    resolved_out = Path(output_path).resolve()
    resolved_in = Path(input_path).resolve()
    if resolved_in.is_file() and resolved_out == resolved_in:
        raise ReportWriteError(f"Refusing to overwrite input file '{resolved_in}'.")
    if resolved_in.is_dir():
        has_files = any(p.is_file() for p in resolved_in.iterdir())
        if has_files and (
            resolved_out == resolved_in or resolved_in in resolved_out.parents
        ):
            raise ReportWriteError(
                f"Refusing to write inside raw input directory '{resolved_in}'."
            )
