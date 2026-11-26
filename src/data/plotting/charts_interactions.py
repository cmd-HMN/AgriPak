from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.common.dataset.columns import DatasetSchema, is_identified, resolve_drivers
from src.common.report.plotting import PlotArtifact, qcut_series
from src.data.analysis import CombinationResult
from src.data.plotting.chart_helpers import (
    _barh,
    _baseline,
    _cat,
    _finish,
    _grid,
    _heatmap,
    _num,
    _nunique,
    _PlotSpec,
    _skip,
)


def _qcut(s: Any, q: int = 4) -> pd.Series:
    return qcut_series(cast(pd.Series, s), q=q)


def _get_drivers(schema: DatasetSchema, df: pd.DataFrame | None = None) -> list[str]:
    if df is None:
        return []
    return [c for c in resolve_drivers(df, schema).numeric_columns if is_identified(c)]


def plot_pairwise_risk_heatmaps(
    df: pd.DataFrame,
    schema: DatasetSchema,
    base_rate: float | None,
    min_n: int,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "pairwise_risk_heatmaps",
        "pairwise_risk_heatmaps.png",
        "Pairwise Driver Risk Heatmap",
        "Interaction Analysis",
    )
    target_col = schema.target_column
    driver_x, driver_y = _get_drivers(schema, df=df)
    if (
        df.empty
        or not is_identified(target_col)
        or not is_identified(driver_x)
        or not is_identified(driver_y)
        or driver_x == driver_y
        or any(c not in df.columns for c in (target_col, driver_x, driver_y))
    ):
        return _skip(spec, assets_dir, "Required driver or target columns missing.")
    sub = pd.DataFrame(
        {
            "x": _num(df, driver_x),
            "y": _num(df, driver_y),
            "target": _num(df, target_col),
        }
    ).dropna()
    if len(sub) < min_n or _nunique(sub["x"]) < 2 or _nunique(sub["y"]) < 2:
        return _skip(spec, assets_dir, "Insufficient data for pairwise heatmap.")
    sub["x_bin"] = _qcut(sub["x"]).astype(str)
    sub["y_bin"] = _qcut(sub["y"]).astype(str)
    agg = cast(
        pd.DataFrame,
        sub.groupby(["y_bin", "x_bin"], observed=True)["target"].agg(
            count="count", rate="mean"
        ),
    ).reset_index()
    agg.loc[agg["count"] < max(3, min_n // 5), "rate"] = np.nan
    pivot = agg.pivot(index="y_bin", columns="x_bin", values="rate")
    if pivot.empty:
        return _skip(spec, assets_dir, "Empty pairwise pivot.")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _heatmap(
        ax,
        pivot,
        f"{target_col} Rate: {driver_x} vs {driver_y}",
        "YlOrRd",
        center=base_rate,
        xlabel=driver_x,
        ylabel=driver_y,
    )
    return _finish(fig, spec, assets_dir)


def plot_top_combinations(
    combos: CombinationResult,
    base_rate: float | None,
    target_col: str,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "top_combinations",
        "top_combinations.png",
        "Top Risky & Protective Combinations",
        "Combination Analysis",
    )
    if not is_identified(target_col) or (
        combos.top_risky.empty and combos.top_protective.empty
    ):
        return _skip(spec, assets_dir, "No feature combinations available.")
    panels: list[tuple[pd.DataFrame, str, str]] = []
    if not combos.top_risky.empty:
        panels.append(
            (
                combos.top_risky.head(10).iloc[::-1],
                "#c53030",
                f"Top Risky Combinations ({target_col})",
            )
        )
    if not combos.top_protective.empty:
        panels.append(
            (
                combos.top_protective.head(10).iloc[::-1],
                "#2f855a",
                f"Top Protective Combinations ({target_col})",
            )
        )
    fig, axes = _grid(len(panels), max_cols=len(panels), col_w=7.0, row_h=5.0)
    for ax, (frame, color, panel_title) in zip(axes, panels):
        _barh(
            ax,
            frame["combination"].astype(str),
            frame["target_rate"].to_numpy(),
            color,
            panel_title,
            "Target Rate",
            base_rate,
        )
    return _finish(fig, spec, assets_dir)


def _plot_interaction_panel(
    ax: Axes,
    x_vals: pd.Series,
    b_vals: pd.Series,
    y_vals: pd.Series,
    primary_driver: str,
    bcol: str,
    base_rate: float | None,
) -> None:
    sub = pd.DataFrame({"x": x_vals, "b": b_vals, "y": y_vals}).dropna()
    if len(sub) < 10 or _nunique(sub["x"]) < 2 or _nunique(sub["b"]) < 2:
        ax.set_title(f"{primary_driver} x {bcol} (insufficient)")
        return
    sub["x_bin"] = _qcut(sub["x"])
    grouped = cast(
        pd.Series, sub.groupby(["x_bin", "b"], observed=True)["y"].mean()
    ).reset_index()
    grouped["bin_str"] = grouped["x_bin"].astype(str)
    for flag_val, grp in grouped.groupby("b", observed=True):
        ax.plot(
            grp["bin_str"].astype(str).tolist(),
            np.asarray(grp["y"].to_numpy(), dtype=float),
            marker="o",
            label=f"{bcol}={int(cast(Any, flag_val))}",
        )
    _baseline(ax, base_rate, vertical=False)
    ax.set_title(f"{primary_driver} x {bcol}")
    ax.set_ylabel("Target Rate")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)


def plot_continuous_binary_interactions(
    df: pd.DataFrame,
    schema: DatasetSchema,
    base_rate: float | None,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "continuous_binary_interactions",
        "continuous_binary_interactions.png",
        "Continuous x Binary Interactions",
        "Interaction Analysis",
    )
    target_col = schema.target_column
    primary_driver, _ = _get_drivers(schema, df=df)
    bin_cols = [c for c in schema.binary_columns if c in df.columns and c != target_col]
    if (
        df.empty
        or not is_identified(target_col)
        or not is_identified(primary_driver)
        or primary_driver not in df.columns
        or target_col not in df.columns
        or not bin_cols
    ):
        return _skip(spec, assets_dir, "Required continuous or binary columns missing.")
    show_cols = bin_cols[:4]
    fig, axes = _grid(len(show_cols), max_cols=2, col_w=6.0, row_h=4.0)
    x_vals, y_vals = _num(df, primary_driver), _num(df, target_col)
    for ax, bcol in zip(axes, show_cols):
        _plot_interaction_panel(
            ax, x_vals, _num(df, bcol), y_vals, primary_driver, bcol, base_rate
        )
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_subgroup_risk_surfaces(
    surfaces: dict[str, pd.DataFrame],
    schema: DatasetSchema,
    base_rate: float | None,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "subgroup_risk_surfaces",
        "subgroup_risk_surfaces.png",
        "Subgroup Risk Surfaces",
        "Subgroup Analysis",
    )
    if not surfaces:
        return _skip(spec, assets_dir, "No subgroup surfaces available.")
    keys = list(surfaces.keys())[:6]
    fig, axes = _grid(len(keys), max_cols=3, col_w=5.5, row_h=4.5)
    driver_x, driver_y = _get_drivers(schema)
    for ax, key in zip(axes, keys):
        _heatmap(
            ax,
            surfaces[key],
            f"Subgroup: {key}",
            "YlOrRd",
            center=base_rate,
            xlabel=driver_x,
            ylabel=driver_y,
        )
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_kruskal_wallis_boxplots(
    df: pd.DataFrame,
    schema: DatasetSchema,
    kruskal_df: pd.DataFrame,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "kruskal_wallis_boxplots",
        "kruskal_wallis_boxplots.png",
        "Top Kruskal-Wallis Group Differences",
        "Group Comparisons",
    )
    if df.empty or kruskal_df.empty:
        return _skip(spec, assets_dir, "No Kruskal-Wallis results available.")
    top_rows = kruskal_df.head(6)
    fig, axes = _grid(len(top_rows), max_cols=2, col_w=6.0, row_h=4.0)
    for ax, (_, row) in zip(axes, top_rows.iterrows()):
        num_col, cat_col = str(row["numeric_col"]), str(row["categorical_col"])
        if num_col not in df.columns or cat_col not in df.columns:
            continue
        sub = pd.DataFrame(
            {num_col: _num(df, num_col), cat_col: _cat(df, cat_col)}
        ).dropna(subset=[num_col])
        sns.boxplot(data=sub, x=cat_col, y=num_col, ax=ax)
        ax.set_title(
            f"{num_col} by {cat_col} (eps2={float(cast(Any, row['epsilon_sq'])):.2f})"
        )
        ax.tick_params(axis="x", rotation=25)
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_cramers_v_associations(
    target_v: pd.DataFrame,
    pair_v: pd.DataFrame,
    target_col: str,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "cramers_v_associations",
        "cramers_v_associations.png",
        "Cramers V Associations",
        "Association Strength",
    )
    has_target = not target_v.empty and is_identified(target_col)
    has_pair = not pair_v.empty and pair_v.shape[0] >= 2
    if not has_target and not has_pair:
        return _skip(spec, assets_dir, "No Cramers V tables available.")
    n_panels = int(has_target) + int(has_pair)
    fig, axes = _grid(n_panels, max_cols=n_panels, col_w=6.5, row_h=5.0)
    idx = 0
    if has_target:
        ordered = target_v.sort_values("cramers_v", ascending=True)
        _barh(
            axes[idx],
            ordered["feature"].astype(str),
            ordered["cramers_v"].to_numpy(),
            "#2b6cb0",
            f"Cramers V vs {target_col}",
            "Cramers V",
        )
        idx += 1
    if has_pair:
        _heatmap(
            axes[idx],
            pair_v,
            "Pairwise Cramers V",
            "Blues",
            annot=pair_v.shape[0] <= 12,
            vmin=0.0,
            vmax=1.0,
        )
    return _finish(fig, spec, assets_dir)
