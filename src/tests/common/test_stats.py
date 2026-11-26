import pandas as pd

from src.common.dataset.stats import (
    compute_cramers_v,
    compute_numeric_shape_metrics,
    compute_spearman_with_target,
)


def test_compute_cramers_v() -> None:
    s1 = pd.Series(["a", "a", "b", "b", "c", "c"])
    s2 = pd.Series(["x", "x", "y", "y", "z", "z"])
    assert compute_cramers_v(s1, s2) > 0.9
    const = pd.Series(["a", "a", "a", "a", "a", "a"])
    assert compute_cramers_v(const, s2) == 0.0


def test_compute_numeric_shape_metrics(clean_df: pd.DataFrame) -> None:
    metrics = compute_numeric_shape_metrics(clean_df, ["area_ha", "yield_t_ha"])
    assert len(metrics) == 2
    assert set(metrics["column"]) == {"area_ha", "yield_t_ha"}
    assert "skewness" in metrics.columns
    assert "kurtosis" in metrics.columns


def test_compute_spearman_with_target(clean_df: pd.DataFrame) -> None:
    df = pd.concat([clean_df, clean_df], ignore_index=True)
    target = pd.Series(df["yield_t_ha"])
    res = compute_spearman_with_target(df, ["area_ha", "production_t"], target)
    assert len(res) == 2
    assert "spearman_rho" in res.columns
    assert "abs_spearman" in res.columns
