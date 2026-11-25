from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.plotting.helpers import (
    PlotArtifact,
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
from src.data.schema import DatasetSchema
from src.data.utils import is_identified


def plot_target_distribution(
    df: pd.DataFrame, schema: DatasetSchema, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "target_distribution",
        "target_distribution.png",
        "Target Distribution",
        "Target Overview",
    )
    target_col = schema.target_column
    if df.empty or not is_identified(target_col) or target_col not in df.columns:
        return _skip(spec, assets_dir, "Target column unavailable.")
    series = cast(pd.Series, df[target_col].dropna())
    if series.empty:
        return _skip(spec, assets_dir, "Target column has no valid values.")
    counts = series.value_counts().sort_index()
    values = np.asarray(counts.to_numpy(), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(k) for k in counts.index], values, color="#2b6cb0")
    total = float(values.sum())
    for bar, cnt in zip(bars, values):
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(cnt)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"{spec[2]}: {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    return _finish(fig, spec, assets_dir)


def plot_numeric_distributions(
    df: pd.DataFrame, schema: DatasetSchema, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "numeric_distributions",
        "numeric_distributions.png",
        "Numeric Feature Distributions",
        "Univariate Distributions",
    )
    cols = [c for c in schema.numeric_columns if c in df.columns]
    if df.empty or not cols:
        return _skip(spec, assets_dir, "No numeric columns available.")
    fig, axes = _grid(len(cols))
    for ax, col in zip(axes, cols):
        vals = _num(df, col).dropna()
        if vals.empty:
            ax.set_title(f"{col} (empty)")
            continue
        sns.histplot(cast(Any, vals), kde=_nunique(vals) > 5, ax=ax, color="#3182ce")
        ax.set_title(col)
        ax.set_xlabel(col)
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_categorical_counts(
    df: pd.DataFrame, schema: DatasetSchema, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "categorical_counts",
        "categorical_counts.png",
        "Categorical Feature Counts",
        "Univariate Distributions",
    )
    cols = [c for c in schema.categorical if c in df.columns]
    if df.empty or not cols:
        return _skip(spec, assets_dir, "No categorical columns available.")
    fig, axes = _grid(len(cols), max_cols=2, col_w=6.0, row_h=4.0)
    for ax, col in zip(axes, cols):
        counts = _cat(df, col).value_counts().head(15)
        _barh(
            ax,
            list(counts.index)[::-1],
            counts.to_numpy()[::-1],
            "#2f855a",
            col,
            "Count",
        )
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_binary_flag_counts(
    df: pd.DataFrame, schema: DatasetSchema, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "binary_flag_counts",
        "binary_flag_counts.png",
        "Binary Signal Prevalence",
        "Univariate Distributions",
    )
    cols = [
        c
        for c in schema.binary_columns
        if c in df.columns and c != schema.target_column
    ]
    if df.empty or not cols:
        return _skip(spec, assets_dir, "No binary flag columns available.")
    rates = sorted(
        [
            (
                c,
                float((_num(df, c).dropna() == 1).mean() * 100.0)
                if not _num(df, c).dropna().empty
                else 0.0,
            )
            for c in cols
        ],
        key=lambda item: item[1],
    )
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(rates) + 1.5)))
    _barh(
        ax,
        [r[0] for r in rates],
        [r[1] for r in rates],
        "#805ad5",
        spec[2],
        "Prevalence (%)",
    )
    return _finish(fig, spec, assets_dir)


def plot_risk_rate_by_category(
    df: pd.DataFrame,
    schema: DatasetSchema,
    base_rate: float | None,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "risk_rate_by_category",
        "risk_rate_by_category.png",
        "Target Rate by Category",
        "Bivariate Relationships",
    )
    target_col = schema.target_column
    cols = [
        c
        for c in list(schema.categorical) + list(schema.binary_columns)
        if c in df.columns and c != target_col
    ]
    if (
        df.empty
        or not is_identified(target_col)
        or target_col not in df.columns
        or not cols
    ):
        return _skip(spec, assets_dir, "Target or categorical columns missing.")
    target = _num(df, target_col)
    if target.dropna().empty:
        return _skip(spec, assets_dir, "Target has no numeric values.")
    fig, axes = _grid(len(cols), max_cols=2, col_w=6.2, row_h=4.0)
    for ax, col in zip(axes, cols):
        temp = pd.DataFrame({"cat": _cat(df, col), "target": target}).dropna(
            subset=["target"]
        )
        grouped = (
            cast(pd.Series, temp.groupby("cat", observed=True)["target"].mean())
            .sort_values(ascending=True)
            .tail(15)
        )
        _barh(
            ax,
            [str(k) for k in grouped.index],
            grouped.to_numpy(),
            "#dd6b20",
            f"{target_col} Rate by {col}",
            "Target Rate",
            base_rate,
        )
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_numeric_by_target(
    df: pd.DataFrame, schema: DatasetSchema, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "numeric_by_target",
        "numeric_by_target.png",
        "Numeric Features by Target",
        "Bivariate Relationships",
    )
    target_col = schema.target_column
    cols = [c for c in schema.numeric_columns if c in df.columns and c != target_col]
    if (
        df.empty
        or not is_identified(target_col)
        or target_col not in df.columns
        or not cols
    ):
        return _skip(spec, assets_dir, "Target or numeric columns missing.")
    fig, axes = _grid(len(cols))
    for ax, col in zip(axes, cols):
        sub = pd.DataFrame({col: _num(df, col), target_col: df[target_col]}).dropna()
        if sub.empty or _nunique(sub[target_col]) < 2:
            ax.set_title(f"{col} (insufficient groups)")
            continue
        sns.boxplot(data=sub, x=target_col, y=col, ax=ax, hue=target_col, legend=False)
        ax.set_title(f"{col} by {target_col}")
    return _finish(fig, spec, assets_dir, suptitle=True)


def plot_spearman_correlation(
    spearman_matrix: pd.DataFrame, assets_dir: Path
) -> PlotArtifact:
    spec: _PlotSpec = (
        "spearman_correlation",
        "spearman_correlation.png",
        "Spearman Correlation Matrix",
        "Correlation Structure",
    )
    if spearman_matrix.empty or spearman_matrix.shape[0] < 2:
        return _skip(spec, assets_dir, "Spearman matrix is empty.")
    size = max(6.0, 0.65 * len(spearman_matrix.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _heatmap(
        ax,
        spearman_matrix,
        spec[2],
        "coolwarm",
        annot=len(spearman_matrix.columns) <= 14,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
    )
    return _finish(fig, spec, assets_dir)


def plot_decile_risk_curves(
    decile_curves: dict[str, pd.DataFrame],
    base_rate: float | None,
    target_col: str,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "decile_risk_curves",
        "decile_risk_curves.png",
        "Quantile Target Rate Curves",
        "Non-Linear Risk Curves",
    )
    if not decile_curves or not is_identified(target_col):
        return _skip(spec, assets_dir, "No decile curves available.")
    cols = list(decile_curves.keys())
    fig, axes = _grid(len(cols), col_w=5.2, row_h=3.8)
    for ax, col in zip(axes, cols):
        curve = decile_curves[col]
        bins = np.asarray(curve["bin_index"].to_numpy(), dtype=float)
        ax.plot(
            bins,
            np.asarray(curve["target_rate"].to_numpy(), dtype=float),
            marker="o",
            color="#c53030",
            linewidth=2,
        )
        _baseline(ax, base_rate, vertical=False)
        ax.set_title(f"{col} -> {target_col}")
        ax.set_xlabel("Quantile Bin")
        ax.set_ylabel("Target Rate")
        ax.set_xticks(bins)
    return _finish(fig, spec, assets_dir, suptitle=True)
