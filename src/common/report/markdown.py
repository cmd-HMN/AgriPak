import math
from typing import Any

import numpy as np
import pandas as pd

from src.common.core.errors import EXIT_CODE_TABLE
from src.common.core.validation import is_missing_scalar


def format_float(value: Any, digits: int = 3, default: str = "N/A") -> str:
    if is_missing_scalar(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isinf(numeric):
        return str(numeric)
    return f"{numeric:.{digits}f}"


def format_pct(value: Any, digits: int = 2, scale: float = 100.0) -> str:
    if is_missing_scalar(value):
        return "N/A"
    try:
        return f"{float(value) * scale:.{digits}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_cell(value: object, float_digits: int) -> str:
    if is_missing_scalar(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return str(numeric) if math.isinf(numeric) else f"{numeric:.{float_digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


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
    for idx, row in view.iterrows():
        cells = [_format_cell(idx, float_digits)] if include_index else []
        cells.extend(
            _format_cell(row.iloc[i], float_digits) for i in range(len(view.columns))
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def append_markdown_table(
    lines: list[str],
    title: str,
    df: pd.DataFrame,
    max_rows: int = 12,
    empty_note: str = "_No records available._",
    float_digits: int = 3,
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if df.empty:
        lines.append(empty_note)
    else:
        lines.append(
            dataframe_to_markdown(df, float_digits=float_digits, max_rows=max_rows)
        )
    lines.append("")


def render_plot_markdown(
    plot_map: dict[str, Any],
    plot_id: str,
    asset_prefix: str,
    fallback_title: str,
) -> list[str]:
    artifact = plot_map.get(plot_id)
    if artifact is None:
        return [f"_{fallback_title}: not generated._", ""]
    if (
        not getattr(artifact, "generated", False)
        or getattr(artifact, "path", None) is None
    ):
        reason = getattr(artifact, "skip_reason", "") or "skipped"
        return [
            f"_{getattr(artifact, 'title', fallback_title)}: skipped ({reason})._",
            "",
        ]
    prefix = asset_prefix.rstrip("/")
    filename = artifact.path.name
    rel = f"{prefix}/{filename}" if prefix else filename
    return [f"![{artifact.title}]({rel})", "", f"_{artifact.caption}_", ""]


def render_exit_code_table() -> str:
    df = pd.DataFrame(
        [
            {"code": code, "name": name, "meaning": meaning}
            for code, name, meaning in EXIT_CODE_TABLE
        ]
    )
    return dataframe_to_markdown(df)
