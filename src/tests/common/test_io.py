from pathlib import Path

import pandas as pd
import pytest

from src.common.core.errors import (
    EmptyDataError,
    InputNotFoundError,
    MalformedDataError,
)
from src.common.core.io import (
    ensure_directory,
    load_dataset,
    load_raw_text,
    save_dataframe,
    write_text_file,
)


@pytest.mark.parametrize("ext", ["csv", "tsv", "json", "parquet"])
def test_roundtrip_formats(tmp_path: Path, clean_df: pd.DataFrame, ext: str) -> None:
    if ext == "parquet":
        pytest.importorskip("pyarrow")
    out_file = tmp_path / f"sample.{ext}"
    saved = save_dataframe(clean_df, out_file)
    assert saved == out_file
    loaded = load_dataset(out_file)
    assert loaded.shape == clean_df.shape
    assert list(loaded.columns) == list(clean_df.columns)


def test_load_raw_text_preserves_strings(tmp_path: Path) -> None:
    csv_file = tmp_path / "raw.csv"
    csv_file.write_text("code,val\n001,10\n", encoding="utf-8")
    df = load_raw_text(csv_file)
    assert df.loc[0, "code"] == "001"


def test_load_directory_combines_parts(
    sample_csv_dir: Path, clean_df: pd.DataFrame
) -> None:
    loaded = load_dataset(sample_csv_dir)
    assert len(loaded) == len(clean_df)


def test_io_errors(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(EmptyDataError):
        load_dataset(empty_file)
    bad_ext = tmp_path / "bad.xyz"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(MalformedDataError):
        load_dataset(bad_ext)


def test_ensure_directory_and_write_text_file(tmp_path: Path) -> None:
    target_dir = ensure_directory(tmp_path / "nested" / "dir")
    assert target_dir.is_dir()
    out = write_text_file("hello world", target_dir / "note.md")
    assert out.read_text(encoding="utf-8") == "hello world"
