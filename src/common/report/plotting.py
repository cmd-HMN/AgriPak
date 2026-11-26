from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.common.core.io import ensure_directory


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    filename: str
    title: str
    caption: str


@dataclass(frozen=True)
class PlotArtifact:
    plot_id: str
    title: str
    caption: str
    path: Path | None
    generated: bool
    skip_reason: str = ""


def save_figure(fig: Figure, path: Path) -> Path:
    ensure_directory(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def skip_plot(spec: PlotSpec, reason: str) -> PlotArtifact:
    return PlotArtifact(
        plot_id=spec.plot_id,
        title=spec.title,
        caption=spec.caption,
        path=None,
        generated=False,
        skip_reason=reason,
    )


def finish_plot(spec: PlotSpec, fig: Figure, output_dir: Path) -> PlotArtifact:
    out_path = save_figure(fig, output_dir / spec.filename)
    return PlotArtifact(
        plot_id=spec.plot_id,
        title=spec.title,
        caption=spec.caption,
        path=out_path,
        generated=True,
    )


def create_grid(
    n_panels: int,
    max_cols: int = 2,
    panel_w: float = 5.5,
    panel_h: float = 4.0,
) -> tuple[Figure, list[Axes]]:
    ncols = max(1, min(max_cols, n_panels))
    nrows = max(1, int(np.ceil(n_panels / ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )
    flat = list(axes.ravel())
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat[:n_panels]


def draw_baseline(ax: Axes, value: float, label: str = "Baseline") -> None:
    ax.axvline(value, color="#475569", linestyle="--", linewidth=1.2, label=label)


def draw_barh(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    color: str = "#2563eb",
) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def draw_heatmap(
    ax: Axes,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    return im


def qcut_series(series: pd.Series, q: int = 4) -> pd.Series:
    valid = series.dropna()
    if valid.empty or int(valid.nunique()) < 2:
        return pd.Series(index=series.index, dtype="object")
    return cast(
        pd.Series, pd.qcut(valid, q=min(q, int(valid.nunique())), duplicates="drop")
    )


def run_plot_jobs(
    jobs: Sequence[tuple[PlotSpec, Callable[[], PlotArtifact]]],
    progress_callback: Callable[[str], None] | None = None,
) -> list[PlotArtifact]:
    artifacts: list[PlotArtifact] = []
    for spec, job in jobs:
        try:
            artifacts.append(job())
        except Exception as exc:
            artifacts.append(skip_plot(spec, f"plot failed: {exc}"))
        if progress_callback is not None:
            progress_callback(spec.title)
    return artifacts
