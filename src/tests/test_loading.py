from pathlib import Path

import pandas as pd
import pytest

from src.data.errors import (
    EmptyDataError,
    InputNotFoundError,
    InvalidArgumentError,
    MalformedDataError,
)
from src.data.loading import load_dataset, load_raw_text, save_dataframe


def test_csv_tsv_json_roundtrip(synthetic_clean_df: pd.DataFrame, tmp_path: Path) -> None:
    small_df = synthetic_clean_df.head(10).copy()
    for ext in (".csv", ".tsv", ".json"):
        out_path = tmp_path / f"sample{ext}"
        saved = save_dataframe(small_df, out_path)
        assert saved == out_path
        loaded = load_dataset(out_path)
        assert len(loaded) == len(small_df)
        assert list(loaded.columns) == list(small_df.columns)


def test_load_raw_text_preserves_strings(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("col_a,col_b\n001,NA\n002,?\n", encoding="utf-8")
    raw = load_raw_text(csv_path)
    assert raw["col_a"].tolist() == ["001", "002"]
    assert raw["col_b"].tolist() == ["NA", "?"]


def test_load_dataset_from_directory(synthetic_clean_df: pd.DataFrame, tmp_path: Path) -> None:
    sub_dir = tmp_path / "parts"
    sub_dir.mkdir()
    synthetic_clean_df.iloc[:5].to_csv(sub_dir / "part1.csv", index=False)
    synthetic_clean_df.iloc[5:10].to_csv(sub_dir / "part2.csv", index=False)
    combined = load_dataset(sub_dir)
    assert len(combined) == 10


def test_load_dataset_missing_path_and_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(InputNotFoundError):
        load_dataset(empty_dir)


def test_load_dataset_malformed_and_unsupported(tmp_path: Path) -> None:
    bad_ext = tmp_path / "data.txt"
    bad_ext.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(MalformedDataError):
        load_dataset(bad_ext)

    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MalformedDataError):
        load_dataset(bad_json)

    with pytest.raises(InvalidArgumentError):
        save_dataframe(pd.DataFrame({"a": [1]}), tmp_path / "out.txt")


def test_load_dataset_empty_files(tmp_path: Path) -> None:
    zero_byte = tmp_path / "zero.csv"
    zero_byte.write_bytes(b"")
    with pytest.raises(EmptyDataError):
        load_dataset(zero_byte)

    header_only = tmp_path / "header_only.csv"
    header_only.write_text("col_a,col_b\n", encoding="utf-8")
    with pytest.raises(EmptyDataError):
        load_dataset(header_only)
