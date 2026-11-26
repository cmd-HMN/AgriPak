from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def prep_df() -> pd.DataFrame:
    rows = []
    for idx in range(30):
        rows.append(
            {
                "crop": ["Wheat", "Rice", "Cotton"][idx % 3],
                "province": ["Punjab", "Sindh", "KP"][idx % 3],
                "season": "Rabi" if idx % 2 == 0 else "Kharif",
                "year": 2020 + (idx % 5),
                "area_ha": 10.0 + idx * 1.2,
                "yield_t_ha": 2.0 + (idx % 5) * 0.4,
                "production_t": (10.0 + idx * 1.2) * (2.0 + (idx % 5) * 0.4),
                "irrigated": idx % 2,
                "high_risk": 1 if idx % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def prep_csv_path(tmp_path: Path, prep_df: pd.DataFrame) -> Path:
    path = tmp_path / "prep.csv"
    prep_df.to_csv(path, index=False)
    return path
