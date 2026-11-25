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

from src.data.analysis import (
    AnalysisBundle,
    CombinationResult,
    FeatureEngineeringResult,
)
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
        sub = pd.DataFrame({"x": x_vals, "b": _num(df, bcol), "y": y_vals}).dropna()
        if len(sub) < 10 or _nunique(sub["x"]) < 2 or _nunique(sub["b"]) < 2:
            ax.set_title(f"{primary_driver} x {bcol} (insufficient)")
            continue
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


def plot_engineered_features(
    fe_result: FeatureEngineeringResult,
    schema: DatasetSchema,
    base_rate: float | None,
    assets_dir: Path,
) -> PlotArtifact:
    spec: _PlotSpec = (
        "engineered_features",
        "engineered_features.png",
        "Engineered Features & Correlation Comparison",
        "Feature Engineering",
    )
    if fe_result.data.empty or not fe_result.engineered_columns:
        return _skip(spec, assets_dir, "No engineered features generated.")

    has_corr = not fe_result.correlation_comparison.empty
    n_panels = 1 + int(has_corr)
    fig, axes = _grid(n_panels, max_cols=n_panels, col_w=6.5, row_h=4.5)
    first_eng = fe_result.engineered_columns[0]
    sns.histplot(
        cast(Any, _num(fe_result.data, first_eng).dropna()), ax=axes[0], color="#38a169"
    )
    axes[0].set_title(f"Distribution: {first_eng}")

    if has_corr:
        comp = fe_result.correlation_comparison.sort_values(
            "abs_spearman_r", ascending=True
        )
        colors = [
            "#38a169" if bool(flag) else "#718096"
            for flag in comp["is_engineered"].tolist()
        ]
        _barh(
            axes[1],
            comp["feature"].astype(str),
            comp["abs_spearman_r"].to_numpy(),
            colors,
            f"Abs Spearman vs {schema.target_column}",
            "|Spearman r|",
        )
    return _finish(fig, spec, assets_dir)


def generate_all_plots(
    clean_df: pd.DataFrame,
    analysis: AnalysisBundle,
    assets_dir: Path,
) -> list[PlotArtifact]:
    schema = analysis.schema
    base_rate = analysis.base_rate
    return [
        plot_target_distribution(clean_df, schema, assets_dir),
        plot_numeric_distributions(clean_df, schema, assets_dir),
        plot_categorical_counts(clean_df, schema, assets_dir),
        plot_binary_flag_counts(clean_df, schema, assets_dir),
        plot_risk_rate_by_category(clean_df, schema, base_rate, assets_dir),
        plot_numeric_by_target(clean_df, schema, assets_dir),
        plot_spearman_correlation(analysis.spearman_matrix, assets_dir),
        plot_decile_risk_curves(
            analysis.decile_curves, base_rate, schema.target_column, assets_dir
        ),
        plot_pairwise_risk_heatmaps(
            clean_df, schema, base_rate, analysis.min_n, assets_dir
        ),
        plot_top_combinations(
            analysis.combinations, base_rate, schema.target_column, assets_dir
        ),
        plot_continuous_binary_interactions(clean_df, schema, base_rate, assets_dir),
        plot_subgroup_risk_surfaces(
            analysis.subgroup_surfaces, schema, base_rate, assets_dir
        ),
        plot_kruskal_wallis_boxplots(
            clean_df, schema, analysis.kruskal_wallis, assets_dir
        ),
        plot_cramers_v_associations(
            analysis.target_cramers_v,
            analysis.pairwise_cramers_v,
            schema.target_column,
            assets_dir,
        ),
        plot_engineered_features(
            analysis.feature_engineering, schema, base_rate, assets_dir
        ),
    ]
