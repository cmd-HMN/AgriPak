import pandas as pd

from src.data.cleaning import clean_dataset


def test_clean_dataset_deduplicates_and_normalizes() -> None:
    raw = pd.DataFrame(
        {
            "  Crop ": [" wheat ", "wheat ", "RICE", None],
            "Province": ["Punjab", "Punjab", "Sindh", "KP"],
            "Area_HA": ["10.0", "10.0", "12.0", "8.0"],
            "Yield_T_HA": [3.0, 3.0, 2.5, 2.0],
        }
    )
    res = clean_dataset(raw)
    assert len(res.data) < len(raw)
    assert "wheat" in res.data["  Crop "].tolist()
    assert not res.audit.to_dataframe().empty
