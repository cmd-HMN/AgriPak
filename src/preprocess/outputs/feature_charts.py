from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.core.validation import UNIDENTIFIED, as_series, to_numeric_series
from src.common.dataset.columns import extract_binary_target
from src.common.report.plotting import (
    PlotArtifact,
    PlotSpec,
    create_grid,
    finish_plot,
    qcut_series,
    run_plot_jobs,
    skip_plot,
)
from src.preprocess.modeling.workflow import PreprocessBundle

_SPEC_ENG = PlotSpec(
    "engineered_features.png",
    "Engineered Feature Distributions",
    "Mean/Std across engineered signals",
    "Distributions of engineered risk signals",
)
_SPEC_SIG = PlotSpec(
    "signal_sum_risk.png",
    "Target Rate by Binary Signal Sum",
    "Monotonic risk escalation",
    "Positive rate across active binary flags",
)
_SPEC_DRV = PlotSpec(
    "driver_by_signal_curves.png",
    "Primary Driver Risk Curves by Signal Sum",
    "Stratified quantile risk",
    "Primary driver risk curves",
)
_SPEC_CV = PlotSpec(
    "cv_model_comparison.png",
    "5-Fold CV ROC-AUC Comparison",
    "LogisticRegression vs HistGradientBoosting",
    "5-fold CV ROC-AUC comparison",
)


def _plot_engineered_features(bundle: PreprocessBundle, out_dir: Path) -> PlotArtifact:
    cols = [
        c
        for c in bundle.engineered.engineered_cols
        if c in bundle.engineered.df.columns
    ]
    if not cols:
        return skip_plot(_SPEC_ENG, "No engineered columns available")
    fig, axes = create_grid(len(cols))
    for ax, col in zip(axes, cols):
        vals = to_numeric_series(as_series(bundle.engineered.df, col)).dropna()
        ax.hist(vals, bins=25, color="#2b6cb0", alpha=0.85, edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    return finish_plot(_SPEC_ENG, fig, out_dir)


def _plot_signal_sum_risk(bundle: PreprocessBundle, out_dir: Path) -> PlotArtifact:
    target_col = bundle.schema.target_column
    sig_col = bundle.engineered.signal_sum_col
    if (
        not target_col
        or target_col not in bundle.engineered.df.columns
        or sig_col not in bundle.engineered.df.columns
    ):
        return skip_plot(_SPEC_SIG, "Missing target or signal sum column")
    y = extract_binary_target(bundle.engineered.df, target_col)
    grouped = (
        pd.DataFrame({sig_col: bundle.engineered.df[sig_col], "target": y})
        .groupby(sig_col, as_index=False)
        .agg(target=("target", "mean"))
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        grouped[sig_col].astype(str),
        np.asarray(grouped["target"], dtype=float) * 100.0,
        color="#c53030",
        alpha=0.85,
    )
    ax.set_xlabel(sig_col)
    ax.set_ylabel("Positive Rate (%)")
    ax.set_title(_SPEC_SIG.title)
    ax.grid(axis="y", alpha=0.3)
    return finish_plot(_SPEC_SIG, fig, out_dir)


def _plot_driver_by_signal_curves(
    bundle: PreprocessBundle, out_dir: Path
) -> PlotArtifact:
    target_col = bundle.schema.target_column
    driver = bundle.engineered.primary_driver
    sig_col = bundle.engineered.signal_sum_col
    if (
        not target_col
        or driver == UNIDENTIFIED
        or driver not in bundle.engineered.df.columns
    ):
        return skip_plot(_SPEC_DRV, "Missing target or primary driver")
    work = pd.DataFrame(
        {
            "driver_bin": qcut_series(
                to_numeric_series(as_series(bundle.engineered.df, driver)), q=5
            ),
            "sig_bucket": np.where(
                bundle.engineered.df[sig_col] == 0, "0 signals", "1+ signals"
            ),
            "target": extract_binary_target(bundle.engineered.df, target_col),
        }
    ).dropna()
    if work.empty:
        return skip_plot(_SPEC_DRV, "Insufficient data for driver curves")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels: list[str] = []
    for label, sub in work.groupby("sig_bucket"):
        rates = sub.groupby("driver_bin", observed=False, as_index=False).agg(
            target=("target", "mean")
        )
        labels = rates["driver_bin"].astype(str).tolist()
        ax.plot(
            range(len(labels)),
            np.asarray(rates["target"], dtype=float) * 100.0,
            marker="o",
            label=str(label),
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Positive Rate (%)")
    ax.set_title(f"{driver} Risk by {sig_col}")
    ax.legend()
    ax.grid(alpha=0.3)
    return finish_plot(_SPEC_DRV, fig, out_dir)


def _plot_cv_model_comparison(bundle: PreprocessBundle, out_dir: Path) -> PlotArtifact:
    table = bundle.evaluation.cv_table
    if table.empty:
        return skip_plot(_SPEC_CV, "No CV evaluation results")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(table))
    width = 0.35
    ax.bar(
        x - width / 2,
        table["logreg_roc_auc"],
        width,
        label="LogisticRegression",
        color="#3182ce",
    )
    ax.bar(
        x + width / 2,
        table["hgb_roc_auc"],
        width,
        label="HistGradientBoosting",
        color="#2f855a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table["feature_set"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(_SPEC_CV.title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return finish_plot(_SPEC_CV, fig, out_dir)


def generate_preprocess_plots(
    bundle: PreprocessBundle,
    out_dir: Path,
    on_plot: Callable[[str], None] | None = None,
) -> list[PlotArtifact]:
    jobs = [
        (_SPEC_ENG, lambda: _plot_engineered_features(bundle, out_dir)),
        (_SPEC_SIG, lambda: _plot_signal_sum_risk(bundle, out_dir)),
        (_SPEC_DRV, lambda: _plot_driver_by_signal_curves(bundle, out_dir)),
        (_SPEC_CV, lambda: _plot_cv_model_comparison(bundle, out_dir)),
    ]
    return run_plot_jobs(jobs, on_plot)
