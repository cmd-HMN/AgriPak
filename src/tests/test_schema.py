import numpy as np
import pandas as pd
import pytest

from src.data.errors import EmptyDataError, MissingColumnError
from src.data.schema import infer_schema, resolve_drivers
from src.data.utils import UNIDENTIFIED


def test_infer_schema_rich_types(synthetic_clean_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_clean_df)
    assert "num_a" in schema.numeric_continuous
    assert "num_b" in schema.numeric_continuous
    assert "num_c" in schema.numeric_discrete
    assert set(schema.binary_flags) == {"flag_x", "flag_y", "target_bin"}
    assert set(schema.categorical) == {"cat_1", "cat_2"}
    assert schema.target_column in schema.binary_flags
    assert schema.primary_driver in schema.numeric_columns
    assert schema.secondary_driver in schema.numeric_columns
    assert schema.primary_driver != schema.secondary_driver
    assert schema.primary_subgroup in schema.categorical


def test_infer_schema_datetime_and_text(synthetic_datetime_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_datetime_df, max_categorical_cardinality=5)
    assert "event_dt" in schema.datetime_columns
    assert "note_txt" in schema.text_columns
    assert "cat_1" in schema.categorical
    assert schema.target_column == "target_bin"


def test_infer_schema_explicit_target_and_missing(synthetic_clean_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_clean_df, target_column="num_a")
    assert schema.target_column == "num_a"
    assert schema.target_kind == "numeric_continuous"
    assert "num_a" not in schema.all_feature_columns

    with pytest.raises(MissingColumnError):
        infer_schema(synthetic_clean_df, target_column="nonexistent_col")


def test_infer_schema_empty_raises() -> None:
    with pytest.raises(EmptyDataError):
        infer_schema(pd.DataFrame())


def test_infer_schema_unidentified_fallback(synthetic_unidentifiable_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_unidentifiable_df)
    assert set(schema.unidentified_columns) == {"col_a", "col_b"}
    assert schema.target_column == UNIDENTIFIED
    assert schema.target_kind == "unidentified"
    assert schema.primary_driver == UNIDENTIFIED
    assert schema.secondary_driver == UNIDENTIFIED
    assert schema.primary_subgroup == UNIDENTIFIED


def test_resolve_drivers_handles_unidentified_target(synthetic_clean_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_clean_df)
    no_target = resolve_drivers(
        synthetic_clean_df.drop(columns=[schema.target_column]),
        schema,
    )
    assert no_target.primary_driver == UNIDENTIFIED
    assert no_target.secondary_driver == UNIDENTIFIED


def test_infer_schema_numeric_fallback_target_when_no_binary() -> None:
    df = pd.DataFrame(
        {
            "num_a": np.linspace(1.0, 10.0, 20),
            "num_b": np.linspace(10.0, 1.0, 20),
            "cat_1": ["a", "b", "c", "d"] * 5,
        }
    )
    schema = infer_schema(df)
    assert schema.target_column == "num_b"
    assert schema.primary_driver == "num_a"
