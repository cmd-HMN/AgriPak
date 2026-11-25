from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.errors import ReportWriteError
from src.data.utils import (
    UNIDENTIFIED,
    compute_min_subgroup_size,
    configure_logging,
    dataframe_to_markdown,
    ensure_directory,
    get_console,
    is_identified,
    normalize_label,
)


def test_normalize_label_variants() -> None:
    assert normalize_label(None) == UNIDENTIFIED
    assert normalize_label(np.nan) == UNIDENTIFIED
    assert normalize_label("   ") == UNIDENTIFIED
    assert normalize_label("NA") == UNIDENTIFIED
    assert normalize_label("n/a") == UNIDENTIFIED
    assert normalize_label("?") == UNIDENTIFIED
    assert normalize_label("-") == UNIDENTIFIED
    assert normalize_label("  hello   world  ") == "hello world"


def test_is_identified() -> None:
    df = pd.DataFrame({"num_a": [1, 2], "cat_1": ["a", "b"]})
    assert is_identified("num_a") is True
    assert is_identified("num_a", df) is True
    assert is_identified("missing_col", df) is False
    assert is_identified(UNIDENTIFIED, df) is False
    assert is_identified("", df) is False
    assert is_identified(None, df) is False


def test_compute_min_subgroup_size() -> None:
    assert compute_min_subgroup_size(0) == 20
    assert compute_min_subgroup_size(-5) == 20
    assert compute_min_subgroup_size(500) == 20
    assert compute_min_subgroup_size(5000) == 50
    assert compute_min_subgroup_size(100, min_floor=10, fraction=0.05) == 10
    assert compute_min_subgroup_size(1000, min_floor=10, fraction=0.05) == 50


def test_dataframe_to_markdown() -> None:
    empty_df = pd.DataFrame()
    assert "| (empty) |" in dataframe_to_markdown(empty_df)

    df = pd.DataFrame(
        {"a|b": [1, 2, 3], "val": [1.23456, np.nan, 3.0]},
        index=pd.Index(["r1", "r2", "r3"], name="row_id"),
    )
    md = dataframe_to_markdown(df, float_digits=2, max_rows=2, include_index=True)
    lines = md.splitlines()
    assert len(lines) == 4
    assert "row_id" in lines[0]
    assert "a\\|b" in lines[0]
    assert "1.23" in lines[2]


def test_ensure_directory_creates_and_raises_on_file_conflict(tmp_path: Path) -> None:
    new_dir = tmp_path / "a" / "b"
    created = ensure_directory(new_dir)
    assert created.is_dir()

    blocking_file = tmp_path / "blocker"
    blocking_file.write_text("x", encoding="utf-8")
    with pytest.raises(ReportWriteError):
        ensure_directory(blocking_file / "sub")


def test_configure_logging_writes_log_file_and_console(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    logger = configure_logging(debug=True, log_dir=log_dir)
    logger.info("rich logging smoke test")
    for handler in logger.handlers:
        handler.flush()
    log_file = log_dir / "agripak.log"
    assert log_file.exists()
    assert "rich logging smoke test" in log_file.read_text(encoding="utf-8")
    assert get_console(stderr=True).stderr is True
