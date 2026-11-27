from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.core.errors import InvalidArgumentError, TransformError
from src.common.core.io import save_dataframe
from src.common.core.validation import require_non_empty
from src.common.dataset.columns import (
    DatasetSchema,
    extract_binary_target,
    infer_schema,
)


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str
    test_size: float
    random_state: int
    train_positive_rate: float
    test_positive_rate: float
    train_path: Path | None = None
    test_path: Path | None = None


def split_train_test(
    df: pd.DataFrame,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    schema: DatasetSchema | None = None,
) -> SplitResult:
    require_non_empty(df, "df")
    if not (0.0 < float(test_size) < 1.0):
        raise InvalidArgumentError(
            f"test_size must be strictly between 0 and 1, got {test_size}"
        )
    active = schema or infer_schema(df, target_column=target_column)
    target = target_column or active.target_column
    if not target or target not in df.columns:
        raise TransformError(
            "Could not identify target column for stratified train-test split."
        )
    y = extract_binary_target(df, target)
    if y.nunique() < 2 or int(y.value_counts().min()) < 2:
        raise TransformError(
            f"Target column '{target}' needs at least 2 classes with >=2 samples each for stratified split."
        )
    train_raw, test_raw = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    train_df = pd.DataFrame(train_raw).reset_index(drop=True)
    test_df = pd.DataFrame(test_raw).reset_index(drop=True)
    train_rate = float(extract_binary_target(train_df, target).mean())
    test_rate = float(extract_binary_target(test_df, target).mean())
    return SplitResult(
        train_df=train_df,
        test_df=test_df,
        target_column=target,
        test_size=float(test_size),
        random_state=int(random_state),
        train_positive_rate=round(train_rate, 4),
        test_positive_rate=round(test_rate, 4),
    )


def save_train_test_split(
    result: SplitResult,
    output_dir: Path,
    source_input: Path | None = None,
) -> SplitResult:
    out_dir = Path(output_dir)
    train_path = save_dataframe(
        result.train_df, out_dir / "train.csv", source_input=source_input
    )
    test_path = save_dataframe(
        result.test_df, out_dir / "test.csv", source_input=source_input
    )
    return SplitResult(
        train_df=result.train_df,
        test_df=result.test_df,
        target_column=result.target_column,
        test_size=result.test_size,
        random_state=result.random_state,
        train_positive_rate=result.train_positive_rate,
        test_positive_rate=result.test_positive_rate,
        train_path=train_path,
        test_path=test_path,
    )
