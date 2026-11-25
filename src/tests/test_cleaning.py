import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningAudit,
    clean_dataset,
    coerce_column_types,
    finalize_dtypes,
    handle_missing_and_duplicates,
    normalize_text_columns,
    validate_numeric_ranges,
)
from src.data.errors import EmptyDataError
from src.data.schema import infer_schema
from src.data.utils import UNIDENTIFIED


def test_cleaning_steps_individually(synthetic_messy_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_messy_df, target_column="target_bin")
    audit = CleaningAudit()

    step1 = normalize_text_columns(synthetic_messy_df, schema, audit)
    assert step1["cat_1"].iloc[0] == "alpha"
    assert pd.isna(step1["cat_1"].iloc[2])
    assert pd.isna(step1["cat_1"].iloc[6])

    step2 = coerce_column_types(step1, schema, audit)
    assert pd.api.types.is_numeric_dtype(step2["num_b"].dtype)
    assert pd.isna(step2["num_b"].iloc[2])
    assert pd.api.types.is_numeric_dtype(step2["flag_x"].dtype)
    assert step2["flag_x"].iloc[0] == 1.0
    assert step2["flag_x"].iloc[1] == 0.0

    step3 = validate_numeric_ranges(step2, schema, audit)
    assert not np.isinf(step3["num_a"].to_numpy(dtype=float, na_value=np.nan)).any()

    step4 = handle_missing_and_duplicates(step3, schema, audit)
    assert len(step4) < len(step3)
    assert int(step4["target_bin"].isna().sum()) == 0
    assert UNIDENTIFIED in set(step4["cat_1"].astype(str))
    assert int(step4["num_a"].isna().sum()) == 0
    assert int(step4["flag_y"].isna().sum()) == 0

    step5 = finalize_dtypes(step4, schema, audit)
    assert isinstance(step5["cat_1"].dtype, pd.CategoricalDtype)
    assert step5["num_a"].dtype == np.dtype("float64")
    assert step5["num_c"].dtype == np.dtype("int8")
    assert step5["flag_x"].dtype == np.dtype("int8")


def test_clean_dataset_end_to_end_and_audit_markdown(synthetic_messy_df: pd.DataFrame) -> None:
    result = clean_dataset(synthetic_messy_df, target_column="target_bin")
    assert result.rows_before == len(synthetic_messy_df)
    assert result.rows_after < result.rows_before
    assert result.memory_mb_before >= 0.0
    assert result.memory_mb_after >= 0.0
    md = result.audit.to_markdown()
    assert "| step |" in md
    assert "normalize_text" in md


def test_clean_dataset_raises_when_all_targets_missing() -> None:
    df = pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0],
            "target_bin": [np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(EmptyDataError):
        clean_dataset(df, target_column="target_bin")


def test_clean_dataset_raises_on_empty_df() -> None:
    with pytest.raises(EmptyDataError):
        clean_dataset(pd.DataFrame())
