import pandas as pd

from src.common.dataset.columns import infer_schema
from src.preprocess.modeling.evaluation import evaluate_feature_sets
from src.preprocess.modeling.workflow import run_preprocess_pipeline
from src.preprocess.transforms.selection import select_features


def test_run_ablation_study(prep_df: pd.DataFrame) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    schema = infer_schema(df)
    selection = select_features(df, schema=schema)
    ev = evaluate_feature_sets(df, schema=schema, selection=selection, n_splits=3)
    assert not ev.cv_table.empty
    assert "feature_set" in ev.cv_table.columns
    assert "logreg_roc_auc" in ev.cv_table.columns
    assert "hgb_roc_auc" in ev.cv_table.columns


def test_run_preprocessing_pipeline(prep_df: pd.DataFrame) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    result = run_preprocess_pipeline(df, n_splits=3)
    assert len(result.processed_df) == len(df)
    assert not result.evaluation.cv_table.empty
