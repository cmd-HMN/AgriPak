from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    num_a = rng.normal(50.0, 15.0, size=n).round(2)
    num_b = rng.normal(25.0, 5.0, size=n).round(2)
    num_c = rng.integers(1, 6, size=n)
    flag_x = (num_a > 52.0).astype(int)
    flag_y = (rng.random(n) > 0.6).astype(int)
    cat_1 = np.array(["alpha", "beta", "gamma", "delta"])[rng.integers(0, 4, size=n)]
    cat_2 = np.array(["north", "south", "east"])[rng.integers(0, 3, size=n)]
    score = 0.08 * (num_a - 50.0) - 0.06 * (num_b - 25.0) + 0.9 * flag_x + 0.4 * flag_y
    prob = 1.0 / (1.0 + np.exp(-score))
    target_bin = (prob > 0.5).astype(int)
    return pd.DataFrame(
        {
            "num_a": num_a,
            "num_b": num_b,
            "num_c": num_c,
            "flag_x": flag_x,
            "flag_y": flag_y,
            "cat_1": cat_1,
            "cat_2": cat_2,
            "target_bin": target_bin,
        }
    )


@pytest.fixture
def synthetic_messy_df() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "num_a": [10.5, 20.0, np.inf, 40.2, 55.5, 60.1, np.nan, 70.0, 80.5, 90.0],
            "num_b": ["12.0", "15.5", "NA", "22.0", "25.0", "30.0", "35.0", "40.0", "45.0", "50.0"],
            "num_c": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "flag_x": ["Yes", "no", "TRUE", "false", "1", "0", "NA", "y", "n", "yes"],
            "flag_y": [1, 0, 1, 0, 1, 0, 1, 0, np.nan, 1],
            "cat_1": [" alpha ", "beta", "NA", "gamma", "  delta  ", "alpha", "?", "beta", "gamma", "delta"],
            "cat_2": ["north", "south", "east", "north", "south", "east", "north", "south", "east", "north"],
            "target_bin": [0, 1, 0, 1, 0, 1, 0, np.nan, 1, 0],
        }
    )
    dup_row = base.iloc[[0]].copy()
    return pd.concat([base, dup_row], ignore_index=True)


@pytest.fixture
def synthetic_datetime_df() -> pd.DataFrame:
    n = 30
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "event_dt": pd.date_range("2025-01-01", periods=n, freq="D"),
            "num_a": rng.normal(10.0, 2.0, size=n).round(2),
            "cat_1": ["grp_a", "grp_b", "grp_c"] * 10,
            "note_txt": [f"observation record number {i} with extra text" for i in range(n)],
            "target_bin": [0, 1] * 15,
        }
    )


@pytest.fixture
def synthetic_unidentifiable_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col_a": ["NA", "?", "", "none", "null", "-"],
            "col_b": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )


@pytest.fixture
def synthetic_csv_path(synthetic_clean_df: pd.DataFrame, tmp_path: Path) -> Path:
    csv_file = tmp_path / "synthetic.csv"
    synthetic_clean_df.to_csv(csv_file, index=False)
    return csv_file
