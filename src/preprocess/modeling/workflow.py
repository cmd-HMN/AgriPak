from dataclasses import dataclass

import pandas as pd

from src.common.dataset.columns import DatasetSchema, infer_schema
from src.common.core.validation import require_non_empty
from src.preprocess.transforms.encoders import (
    EncodingResult,
    encode_categorical_columns,
)
from src.preprocess.modeling.evaluation import EvaluationResult, evaluate_feature_sets
from src.preprocess.transforms.scalers import ScalingResult, scale_numeric_columns
from src.preprocess.transforms.selection import FeatureSelectionResult, select_features
from src.preprocess.transforms.transformers import (
    EngineeredFeaturesResult,
    engineer_features,
)


@dataclass(frozen=True)
class PreprocessBundle:
    raw_df: pd.DataFrame
    schema: DatasetSchema
    engineered: EngineeredFeaturesResult
    selection: FeatureSelectionResult
    encoding: EncodingResult
    scaling: ScalingResult
    evaluation: EvaluationResult
    processed_df: pd.DataFrame


def run_preprocess_pipeline(
    df: pd.DataFrame,
    scale_method: str = "standard",
    encode_method: str = "onehot",
    min_spearman: float = 0.03,
    min_cramers_v: float = 0.05,
    n_splits: int = 5,
    run_cv: bool = True,
) -> PreprocessBundle:
    require_non_empty(df, "df")
    schema = infer_schema(df)
    engineered = engineer_features(df, schema)
    selection = select_features(
        engineered.df, schema, engineered.engineered_cols, min_spearman, min_cramers_v
    )
    encoding = encode_categorical_columns(
        engineered.df, selection.selected_categorical, method=encode_method
    )
    num_to_scale = selection.selected_numeric + selection.selected_engineered
    scaling = scale_numeric_columns(encoding.df, num_to_scale, method=scale_method)
    evaluation = (
        evaluate_feature_sets(engineered.df, schema, selection, n_splits=n_splits)
        if run_cv
        else evaluate_feature_sets(engineered.df, schema, None, n_splits=n_splits)
    )
    return PreprocessBundle(
        raw_df=df.copy(),
        schema=schema,
        engineered=engineered,
        selection=selection,
        encoding=encoding,
        scaling=scaling,
        evaluation=evaluation,
        processed_df=scaling.df,
    )
