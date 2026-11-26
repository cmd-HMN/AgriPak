from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.core.errors import (
    EmptyDataError,
    InvalidArgumentError,
    MissingColumnError,
    ReportWriteError,
)
from src.common.core.validation import (
    UNIDENTIFIED,
    compute_min_subgroup_size,
    ensure_not_overwriting_input,
    is_identified,
    normalize_label,
    require_columns,
    require_non_empty,
    validate_min_int,
    validate_unit_interval,
)


def test_normalize_label() -> None:
    assert normalize_label("  wheat_grain  ") == "wheat_grain"
    assert normalize_label("rice-paddy") == "rice-paddy"
    assert normalize_label(None) == UNIDENTIFIED
    assert normalize_label("") == UNIDENTIFIED


def test_is_identified(clean_df: pd.DataFrame) -> None:
    assert is_identified("crop", clean_df) is True
    assert is_identified("missing", clean_df) is False
    assert is_identified(UNIDENTIFIED, clean_df) is False
    assert is_identified(None) is False
    assert is_identified("crop") is True


def test_require_non_empty_and_columns(clean_df: pd.DataFrame) -> None:
    assert require_non_empty(clean_df).equals(clean_df)
    with pytest.raises(EmptyDataError):
        require_non_empty(pd.DataFrame(), context="test")
    require_columns(clean_df, ["crop", "yield_t_ha"])
    with pytest.raises(MissingColumnError):
        require_columns(clean_df, ["crop", "nonexistent"])


def test_validate_min_int_and_unit_interval() -> None:
    assert validate_min_int(3, 1, "k") == 3
    with pytest.raises(InvalidArgumentError):
        validate_min_int(0, 1, "k")
    assert validate_unit_interval(0.5, "alpha") == 0.5
    with pytest.raises(InvalidArgumentError):
        validate_unit_interval(1.5, "alpha")
    with pytest.raises(InvalidArgumentError):
        validate_unit_interval(np.nan, "alpha")


def test_compute_min_subgroup_size() -> None:
    assert compute_min_subgroup_size(100) >= 5
    assert compute_min_subgroup_size(10000) == 100
    assert compute_min_subgroup_size(10) >= 5


def test_ensure_not_overwriting_input(tmp_path: Path) -> None:
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("x\n1\n", encoding="utf-8")
    ensure_not_overwriting_input(f1, f2)
    with pytest.raises(ReportWriteError):
        ensure_not_overwriting_input(f1, f1)
