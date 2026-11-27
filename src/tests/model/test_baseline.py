import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.common.core.errors import InputNotFoundError
from src.model.baseline.logistic import (
    load_split_folder,
    run_baseline,
    train_evaluate_logistic,
)
from src.preprocess.modeling.splitting import save_train_test_split, split_train_test


def _make_sample_df() -> pd.DataFrame:
    rows = []
    for idx in range(60):
        low = 1 if idx % 3 == 0 else 0
        rows.append(
            {
                "rainfall_mm": 250.0 + idx * 3.0 if low else 750.0 + idx * 2.0,
                "temperature_c": 22.0 + (idx % 10),
                "fertilizer_used": 0 if low else 1,
                "irrigation_used": 0 if idx % 4 == 0 else 1,
                "crop": ["Wheat", "Rice", "Cotton"][idx % 3],
                "low_yield_risk": low,
            }
        )
    return pd.DataFrame(rows)


def test_train_evaluate_logistic_and_run_baseline(tmp_path: Path) -> None:
    df = _make_sample_df()
    split = save_train_test_split(
        split_train_test(
            df, target_column="low_yield_risk", test_size=0.2, random_state=42
        ),
        tmp_path / "splits",
    )
    metrics = train_evaluate_logistic(
        split.train_df, split.test_df, target_column="low_yield_risk", random_state=42
    )
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert metrics.train_rows == 48
    assert metrics.test_rows == 12

    from_disk = run_baseline(tmp_path / "splits", target_column="low_yield_risk")
    assert from_disk.f1 == metrics.f1


def test_load_split_folder_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(InputNotFoundError):
        load_split_folder(tmp_path / "missing_dir")


def test_model_cli_baseline_and_missing_exit_code(tmp_path: Path) -> None:
    df = _make_sample_df()
    splits_dir = tmp_path / "splits"
    save_train_test_split(
        split_train_test(
            df, target_column="low_yield_risk", test_size=0.2, random_state=42
        ),
        splits_dir,
    )
    metrics_json = tmp_path / "metrics.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.model",
            "baseline",
            "--input",
            str(splits_dir),
            "--output",
            str(metrics_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert metrics_json.is_file()

    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.model",
            "baseline",
            "--input",
            str(tmp_path / "nope"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 3
