from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "crop": [
                "Wheat",
                "Rice",
                "Wheat",
                "Cotton",
                "Rice",
                "Wheat",
                "Cotton",
                "Rice",
            ],
            "province": [
                "Punjab",
                "Sindh",
                "Punjab",
                "Punjab",
                "Sindh",
                "KP",
                "KP",
                "Punjab",
            ],
            "season": [
                "Rabi",
                "Kharif",
                "Rabi",
                "Kharif",
                "Kharif",
                "Rabi",
                "Kharif",
                "Rabi",
            ],
            "year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "area_ha": [10.0, 12.5, 15.0, 8.0, 11.0, 14.0, 9.5, 13.0],
            "yield_t_ha": [3.2, 2.8, 3.5, 2.1, 2.9, 3.4, 2.3, 3.0],
            "production_t": [32.0, 35.0, 52.5, 16.8, 31.9, 47.6, 21.85, 39.0],
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "  Crop Name ": [" wheat ", "RICE", None, "wheat ", "Cotton", ""],
            "Province": ["Punjab", "sindh", "Punjab", "Punjab", "KP", "KP"],
            "Season": ["Rabi", "Kharif", "Rabi", "Rabi", "Kharif", "Rabi"],
            "Year": ["2021", "2022", "bad", "2021", "2023", "2024"],
            "Area_HA": ["10.5", "12.0", "15.0", "10.5", "-3.0", "8.0"],
            "Yield_T_HA": [3.1, None, 2.9, 3.1, 2.2, 2.5],
            "Production_T": [32.55, 30.0, 43.5, 32.55, 10.0, 20.0],
        }
    )


@pytest.fixture
def datetime_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_date": ["2022-01-15", "2022-06-20", "2023-03-10", "2023-11-05"],
            "crop": ["Wheat", "Rice", "Wheat", "Cotton"],
            "area": [10.0, 12.0, 14.0, 9.0],
            "yield": [3.0, 2.5, 3.4, 2.2],
        }
    )


@pytest.fixture
def unidentified_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alpha": [1, 2, 3, 4],
            "beta": ["a", "b", "c", "d"],
            "gamma": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def sample_csv_dir(tmp_path: Path, clean_df: pd.DataFrame) -> Path:
    data_dir = tmp_path / "csv_dir"
    data_dir.mkdir()
    clean_df.iloc[:4].to_csv(data_dir / "part1.csv", index=False)
    clean_df.iloc[4:].to_csv(data_dir / "part2.csv", index=False)
    return data_dir
