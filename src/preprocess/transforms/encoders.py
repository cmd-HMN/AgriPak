from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.common.core.errors import TransformError
from src.common.core.validation import as_series, require_non_empty
from src.common.dataset.columns import series_to_text


@dataclass(frozen=True)
class EncodingResult:
    df: pd.DataFrame
    source_columns: list[str]
    encoded_columns: list[str]
    method: str
    cardinality: dict[str, int]


def _build_onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)


def _build_ordinal() -> OrdinalEncoder:
    return OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64
    )


_ENCODERS: dict[str, Callable[[], Any]] = {
    "onehot": _build_onehot,
    "ordinal": _build_ordinal,
}


def build_categorical_encoder(method: str = "onehot") -> Any:
    key = method.strip().lower()
    if key not in _ENCODERS:
        raise TransformError(
            f"Unsupported encoding method: {method}. Available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[key]()


def _prepare_categorical_frame(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.DataFrame(
        {c: series_to_text(as_series(df, c)) for c in cols}, index=df.index
    )
    cardinality = {c: int(as_series(frame, c).nunique(dropna=False)) for c in cols}
    return frame, cardinality


def encode_categorical_columns(
    df: pd.DataFrame, columns: list[str], method: str = "onehot"
) -> EncodingResult:
    require_non_empty(df, "df")
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return EncodingResult(
            df=df.copy(),
            source_columns=[],
            encoded_columns=[],
            method=method,
            cardinality={},
        )
    encoder = build_categorical_encoder(method)
    frame, cardinality = _prepare_categorical_frame(df, cols)
    try:
        transformed = encoder.fit_transform(frame)
    except Exception as exc:
        raise TransformError(
            f"Failed to encode columns {cols} with {method}: {exc}"
        ) from exc
    out_names = (
        list(encoder.get_feature_names_out(cols))
        if hasattr(encoder, "get_feature_names_out")
        else [f"{c}_enc" for c in cols]
    )
    encoded_df = pd.DataFrame(transformed, columns=out_names, index=df.index)
    merged = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    return EncodingResult(
        df=merged,
        source_columns=cols,
        encoded_columns=out_names,
        method=method.strip().lower(),
        cardinality=cardinality,
    )
