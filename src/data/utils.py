import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler

from src.data.errors import ReportWriteError

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
    if _is_missing_scalar(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if _is_missing_scalar(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _format_cell(value: object, float_digits: int) -> str:
    if _is_missing_scalar(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return str(numeric) if math.isinf(numeric) else f"{numeric:.{float_digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def get_console(stderr: bool = False) -> Console:
    return Console(stderr=stderr)


def configure_logging(
    debug: bool = False, log_dir: Path | None = Path("log")
) -> logging.Logger:
    logger = logging.getLogger("agripak.data")
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    rich_handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=debug,
        show_path=debug,
        markup=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)
    if log_dir is not None:
        target_dir = ensure_directory(Path(log_dir))
        file_handler = logging.FileHandler(
            target_dir / "agripak.log", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


def ensure_directory(path: Path) -> Path:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportWriteError(f"Unable to create directory '{target}': {exc}") from exc
    return target


def normalize_label(
    value: object,
    missing_tokens: frozenset[str] = DEFAULT_MISSING_TOKENS,
) -> str:
    if _is_missing_scalar(value):
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


def dataframe_to_markdown(
    df: pd.DataFrame,
    float_digits: int = 3,
    max_rows: int | None = None,
    include_index: bool = False,
) -> str:
    if len(df.columns) == 0 and not include_index:
        return "| (empty) |\n| --- |"

    view = df.head(max_rows) if max_rows is not None and max_rows >= 0 else df
    headers: list[str] = []
    if include_index:
        index_name = view.index.name if view.index.name is not None else "index"
        headers.append(_format_cell(index_name, float_digits))
    headers.extend(_format_cell(col, float_digits) for col in view.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for idx_val, row in view.iterrows():
        cells: list[str] = (
            [_format_cell(idx_val, float_digits)] if include_index else []
        )
        cells.extend(_format_cell(val, float_digits) for val in row.tolist())
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
