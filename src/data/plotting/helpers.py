from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data.errors import ReportWriteError
from src.data.schema import DatasetSchema, resolve_drivers
from src.data.utils import UNIDENTIFIED, is_identified

_PlotSpec = tuple[str, str, str, str]


@dataclass(frozen=True)
class PlotArtifact:
    plot_id: str
    filename: str
    path: Path
    title: str
    section: str
    generated: bool
    note: str = ""


def _save_figure(fig: Figure, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise ReportWriteError(f"Failed to write plot to {output_path}: {exc}") from exc
    finally:
        plt.close(fig)


def _skip(spec: _PlotSpec, assets_dir: Path, reason: str) -> PlotArtifact:
    plot_id, filename, title, section = spec
    return PlotArtifact(
        plot_id,
        filename,
        Path(assets_dir) / filename,
        title,
        section,
        False,
        f"Skipped: {reason}",
    )


def _finish(
    fig: Figure, spec: _PlotSpec, assets_dir: Path, suptitle: bool = False
) -> PlotArtifact:
    plot_id, filename, title, section = spec
    if suptitle:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path = Path(assets_dir) / filename
    _save_figure(fig, out_path)
    return PlotArtifact(plot_id, filename, out_path, title, section, True)


def _grid(
    n_items: int, max_cols: int = 3, col_w: float = 5.0, row_h: float = 3.8
) -> tuple[Figure, list[Axes]]:
    ncols = min(max_cols, max(1, n_items))
    nrows = math.ceil(n_items / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(col_w * ncols, row_h * nrows), squeeze=False
    )
    flat = [cast(Axes, ax) for ax in axes.ravel()]
    for ax in flat[n_items:]:
        ax.set_visible(False)
    return fig, flat[:n_items]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))


def _cat(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, df[col].astype("object").fillna(UNIDENTIFIED).astype(str))


def _nunique(series: Any) -> int:
    return int(cast(Any, cast(pd.Series, series).nunique()))


def _qcut(series: Any, max_bins: int = 5) -> pd.Series:
    s = cast(pd.Series, series)
    return cast(pd.Series, pd.qcut(s, q=min(max_bins, _nunique(s)), duplicates="drop"))


def _baseline(ax: Axes, base_rate: float | None, vertical: bool = True) -> None:
    if base_rate is None:
        return
    if vertical:
        ax.axvline(base_rate, color="#1a202c", linestyle="--", linewidth=1.2)
    else:
        ax.axhline(base_rate, color="#4a5568", linestyle="--", linewidth=1.0)


def _barh(
    ax: Axes,
    labels: Any,
    values: Any,
    color: Any,
    title: str,
    xlabel: str,
    base_rate: float | None = None,
) -> None:
    ax.barh(list(labels), np.asarray(values, dtype=float), color=color)
    _baseline(ax, base_rate, vertical=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def _heatmap(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    cmap: str,
    annot: bool = True,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    kwargs: dict[str, Any] = {"annot": annot, "fmt": ".2f", "cmap": cmap, "ax": ax}
    if center is not None:
        kwargs["center"] = center
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    sns.heatmap(data, **kwargs)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _get_drivers(
    schema: DatasetSchema, df: pd.DataFrame | None = None
) -> tuple[str, str]:
    if is_identified(schema.primary_driver) and is_identified(schema.secondary_driver):
        return schema.primary_driver, schema.secondary_driver
    if df is not None:
        resolved = resolve_drivers(df, schema)
        if is_identified(resolved.primary_driver) or is_identified(
            resolved.secondary_driver
        ):
            return resolved.primary_driver, resolved.secondary_driver
    cols = schema.numeric_columns
    return (
        cols[0] if len(cols) > 0 else UNIDENTIFIED,
        cols[1] if len(cols) > 1 else UNIDENTIFIED,
    )
