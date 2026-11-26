import pandas as pd

from src.common.core.validation import UNIDENTIFIED
from src.common.dataset.columns import (
    apply_to_columns,
    get_drivers,
    infer_schema,
    resolve_drivers,
)


def test_infer_schema_on_clean_df(clean_df: pd.DataFrame) -> None:
    schema = infer_schema(clean_df)
    assert "crop" in schema.categorical
    assert "yield_t_ha" in schema.numeric_columns
    assert len(schema.profiles) == len(clean_df.columns)


def test_resolve_drivers_identifies_target_and_subgroup(clean_df: pd.DataFrame) -> None:
    schema = resolve_drivers(clean_df, infer_schema(clean_df))
    assert schema.target_column != UNIDENTIFIED
    assert schema.primary_subgroup != UNIDENTIFIED
    drivers = get_drivers(schema, clean_df)
    assert len(drivers) >= 1


def test_infer_schema_unidentified_fallbacks(unidentified_df: pd.DataFrame) -> None:
    no_cat = pd.DataFrame(unidentified_df.loc[:, ["alpha", "gamma"]])
    schema = resolve_drivers(no_cat, infer_schema(no_cat))
    assert schema.primary_subgroup == UNIDENTIFIED
    assert schema.target_column in {"alpha", "gamma"}


def test_apply_to_columns(clean_df: pd.DataFrame) -> None:
    result = apply_to_columns(
        clean_df, ["crop", "province"], lambda s: s.astype(str).str.upper()
    )
    assert result["crop"].iloc[0] == "WHEAT"
    assert clean_df["crop"].iloc[0] == "Wheat"
