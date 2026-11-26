import pandas as pd

from src.common.dataset.columns import infer_schema
from src.preprocess.modeling.builders import build_column_transformer
from src.preprocess.transforms.selection import select_features


def test_select_features_separates_keep_and_drop(prep_df: pd.DataFrame) -> None:
    df = prep_df.copy()
    df["noise_const"] = 1.0
    schema = infer_schema(df)
    manifest = select_features(df, schema=schema)
    assert "noise_const" in manifest.dropped_features
    assert len(manifest.selected_features) >= 1


def test_build_sklearn_preprocessor(prep_df: pd.DataFrame) -> None:
    import numpy as np

    preprocessor = build_column_transformer(
        ["area_ha", "yield_t_ha"], ["crop", "province"]
    )
    transformed = np.asarray(preprocessor.fit_transform(prep_df))
    assert transformed.shape[0] == len(prep_df)
