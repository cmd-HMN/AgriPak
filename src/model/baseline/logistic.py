from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.core.errors import InputNotFoundError, TransformError
from src.common.core.io import load_dataset
from src.common.core.validation import as_series, require_non_empty, to_numeric_series
from src.common.dataset.columns import (
    DatasetSchema,
    extract_binary_target,
    infer_schema,
    series_to_text,
)


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_rows: int
    test_rows: int
    n_features: int
    target_column: str
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


def load_split_folder(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(splits_dir)
    train_path = base / "train.csv"
    test_path = base / "test.csv"
    if not train_path.is_file():
        raise InputNotFoundError(f"Training split file not found: '{train_path}'")
    if not test_path.is_file():
        raise InputNotFoundError(f"Test split file not found: '{test_path}'")
    return load_dataset(train_path), load_dataset(test_path)


def _prepare_feature_frame(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    cols = [c for c in num_cols + cat_cols if c in df.columns]
    work = df.loc[:, cols].copy()
    for col in num_cols:
        if col in work.columns:
            work[col] = to_numeric_series(as_series(work, col))
    for col in cat_cols:
        if col in work.columns:
            work[col] = series_to_text(as_series(work, col))
    return work


def _build_baseline_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int,
) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise TransformError("No usable numeric or categorical features found.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=int(random_state)),
            ),
        ]
    )


def _feature_lists(
    df: pd.DataFrame,
    schema: DatasetSchema,
    target: str,
) -> tuple[list[str], list[str]]:
    num_cols = [
        c
        for c in list(schema.numeric_columns) + list(schema.binary_columns)
        if c in df.columns and c != target
    ]
    cat_cols = [c for c in list(schema.categorical) if c in df.columns and c != target]
    return list(dict.fromkeys(num_cols)), list(dict.fromkeys(cat_cols))


def train_evaluate_logistic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str | None = None,
    random_state: int = 42,
) -> BaselineMetrics:
    require_non_empty(train_df, "train_df")
    require_non_empty(test_df, "test_df")
    schema = infer_schema(train_df, target_column=target_column)
    target = target_column or schema.target_column
    if not target or target not in train_df.columns or target not in test_df.columns:
        raise TransformError(
            "Target column could not be identified in both train and test splits."
        )
    y_train = extract_binary_target(train_df, target)
    y_test = extract_binary_target(test_df, target)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise TransformError(
            f"Target column '{target}' must contain both classes in train and test splits."
        )
    num_cols, cat_cols = _feature_lists(train_df, schema, target)
    common_num = [c for c in num_cols if c in test_df.columns]
    common_cat = [c for c in cat_cols if c in test_df.columns]
    x_train = _prepare_feature_frame(train_df, common_num, common_cat)
    x_test = _prepare_feature_frame(test_df, common_num, common_cat)
    pipe = _build_baseline_pipeline(common_num, common_cat, random_state=random_state)
    pipe.fit(x_train, y_train)
    y_pred = np.asarray(pipe.predict(x_test), dtype=int)
    y_prob = np.asarray(pipe.predict_proba(x_test), dtype=float)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return BaselineMetrics(
        accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
        precision=round(
            float(precision_score(y_test, y_pred, zero_division=cast(Any, 0))), 4
        ),
        recall=round(
            float(recall_score(y_test, y_pred, zero_division=cast(Any, 0))), 4
        ),
        f1=round(float(f1_score(y_test, y_pred, zero_division=cast(Any, 0))), 4),
        roc_auc=round(float(roc_auc_score(y_test, y_prob)), 4),
        train_rows=len(train_df),
        test_rows=len(test_df),
        n_features=len(common_num) + len(common_cat),
        target_column=target,
        confusion_matrix=(
            (int(cm[0, 0]), int(cm[0, 1])),
            (int(cm[1, 0]), int(cm[1, 1])),
        ),
    )


def run_baseline(
    splits_dir: Path,
    target_column: str | None = None,
    random_state: int = 42,
) -> BaselineMetrics:
    train_df, test_df = load_split_folder(splits_dir)
    return train_evaluate_logistic(
        train_df,
        test_df,
        target_column=target_column,
        random_state=random_state,
    )
