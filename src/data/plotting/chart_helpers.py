from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.common.report.plotting import (
    PlotArtifact,
    PlotSpec,
    create_grid,
    draw_barh,
    draw_baseline,
    draw_heatmap,
    finish_plot,
    skip_plot,
)

_PlotSpec = tuple[str, str, str, str]


def _skip(spec: _PlotSpec, assets_dir: Path, reason: str) -> PlotArtifact:
    return skip_plot(PlotSpec(*spec), reason)


def _finish(
    fig: Any, spec: _PlotSpec, assets_dir: Path, suptitle: bool = False
) -> PlotArtifact:
    if suptitle:
        fig.suptitle(spec[2])
    return finish_plot(PlotSpec(*spec), fig, assets_dir)


def _grid(
    n_panels: int, max_cols: int = 2, col_w: float = 6.0, row_h: float = 4.2
) -> tuple[Any, list[Any]]:
    return create_grid(n_panels, max_cols=max_cols)


def _baseline(
    ax: Any, value: float | None, vertical: bool = False, label: str = "Baseline"
) -> None:
    if value is None:
        return
    if vertical:
        ax.axvline(value, color="#e53e3e", linestyle="--", linewidth=1.2, label=label)
    else:
        draw_baseline(ax, value, label=label)


def _barh(
    ax: Any,
    labels: Any,
    values: Any,
    color: Any,
    title: str,
    xlabel: str,
    baseline: float | None = None,
) -> None:
    draw_barh(
        ax,
        [str(x) for x in labels],
        [float(x) for x in values],
        title,
        xlabel,
        color=color,
    )
    if baseline is not None:
        _baseline(ax, baseline, vertical=True)


def _heatmap(
    ax: Any,
    data: pd.DataFrame,
    title: str,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    fmt: str = ".2f",
    annot: bool = True,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    draw_heatmap(
        ax,
        data.to_numpy(dtype=float),
        [str(c) for c in data.columns],
        [str(i) for i in data.index],
        title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))


def _cat(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, df[col]).astype("string").fillna("Missing")


def _nunique(s: Any) -> int:
    return int(cast(pd.Series, s).dropna().nunique())
