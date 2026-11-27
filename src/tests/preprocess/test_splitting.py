import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.common.core.errors import InvalidArgumentError, TransformError
from src.preprocess.modeling.splitting import save_train_test_split, split_train_test


def test_split_train_test_stratified_and_reproducible(prep_df: pd.DataFrame) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    res_a = split_train_test(df, test_size=0.2, random_state=42)
    res_b = split_train_test(df, test_size=0.2, random_state=42)
    assert len(res_a.train_df) == int(len(df) * 0.8)
    assert len(res_a.test_df) == len(df) - len(res_a.train_df)
    assert res_a.train_df.equals(res_b.train_df)
    assert res_a.test_df.equals(res_b.test_df)
    assert abs(res_a.train_positive_rate - res_a.test_positive_rate) < 0.05


def test_split_train_test_validation_errors(prep_df: pd.DataFrame) -> None:
    with pytest.raises(InvalidArgumentError):
        split_train_test(prep_df, test_size=1.2)
    bad_df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1]})
    with pytest.raises(TransformError):
        split_train_test(bad_df, target_column="b")


def test_save_train_test_split_and_cli(
    tmp_path: Path, prep_df: pd.DataFrame, prep_csv_path: Path
) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    res = save_train_test_split(split_train_test(df), tmp_path / "saved")
    assert res.train_path is not None and res.train_path.is_file()
    assert res.test_path is not None and res.test_path.is_file()

    cli_out = tmp_path / "cli_splits"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.preprocess",
            "split",
            "--input",
            str(prep_csv_path),
            "--output",
            str(cli_out),
            "--seed",
            "42",
            "--test-size",
            "0.2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert (cli_out / "train.csv").is_file()
    assert (cli_out / "test.csv").is_file()
