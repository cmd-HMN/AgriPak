from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def eda_df() -> pd.DataFrame:
    rows = []
    for idx in range(24):
        crop = ["Wheat", "Rice", "Cotton"][idx % 3]
        province = ["Punjab", "Sindh", "KP"][idx % 3]
        season = "Rabi" if idx % 2 == 0 else "Kharif"
        rows.append(
            {
                "crop": crop,
                "province": province,
                "season": season,
                "year": 2020 + (idx % 4),
                "area_ha": 10.0 + idx * 1.5,
                "yield_t_ha": 2.0 + (idx % 5) * 0.4,
                "production_t": (10.0 + idx * 1.5) * (2.0 + (idx % 5) * 0.4),
                "high_risk": 1 if idx % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def eda_csv_path(tmp_path: Path, eda_df: pd.DataFrame) -> Path:
    path = tmp_path / "eda.csv"
    eda_df.to_csv(path, index=False)
    return path
