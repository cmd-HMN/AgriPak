import pandas as pd
import pytest

from src.common.core.errors import TransformError
from src.preprocess.transforms.encoders import encode_categorical_columns
from src.preprocess.transforms.scalers import scale_numeric_columns


def test_encode_features_onehot_and_ordinal(prep_df: pd.DataFrame) -> None:
    oh_res = encode_categorical_columns(prep_df, ["crop", "province"], method="onehot")
    assert oh_res.df.shape[1] > prep_df.shape[1] - 2
    ord_res = encode_categorical_columns(
        prep_df, ["crop", "province"], method="ordinal"
    )
    assert pd.api.types.is_numeric_dtype(ord_res.df["crop"])
    with pytest.raises(TransformError):
        encode_categorical_columns(prep_df, ["crop"], method="unknown_method")


def test_scale_features_methods_and_zero_variance(prep_df: pd.DataFrame) -> None:
    df = prep_df.copy()
    df["const_col"] = 5.0
    for method in ("standard", "robust", "minmax"):
        scaled = scale_numeric_columns(df, ["area_ha", "const_col"], method=method)
        assert not bool(scaled.df["const_col"].isna().any())
    with pytest.raises(TransformError):
        scale_numeric_columns(df, ["area_ha"], method="bad_scaler")
