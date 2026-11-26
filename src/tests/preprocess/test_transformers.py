import pandas as pd

from src.preprocess.transforms.transformers import engineer_features


def test_engineer_features_adds_4_columns_and_is_immutable(
    prep_df: pd.DataFrame,
) -> None:
    original_cols = list(prep_df.columns)
    res = engineer_features(prep_df)
    assert list(prep_df.columns) == original_cols
    assert len(res.engineered_cols) == 4
    for col in res.engineered_cols:
        assert col in res.df.columns


def test_engineer_features_unidentified_fallback() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    res = engineer_features(df)
    assert len(res.df) == 4
    assert isinstance(res.engineered_cols, list)
