from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.common.core.validation import (
    as_series,
    require_non_empty,
    to_numeric_series,
)
from src.common.dataset.columns import (
    DatasetSchema,
    extract_binary_target,
    infer_schema,
    resolve_drivers,
    series_to_float,
    series_to_text,
)
from src.preprocess.modeling.builders import build_hgb_pipeline, build_logistic_pipeline
from src.preprocess.transforms.selection import FeatureSelectionResult


@dataclass(frozen=True)
class EvaluationResult:
    cv_table: pd.DataFrame
    ablation_delta: float
    noise_lift: float
    n_splits: int
    feature_sets: dict[str, list[str]]


def _prepare_eval_frame(
    df: pd.DataFrame, num_cols: list[str], bin_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    work = df.copy()
    for col in num_cols:
        if col in work.columns:
            work[col] = to_numeric_series(as_series(work, col))
    for col in bin_cols:
        if col in work.columns:
            work[col] = series_to_float(as_series(work, col))
    for col in cat_cols:
        if col in work.columns:
            work[col] = series_to_text(as_series(work, col))
    return work


def _score_single_model(
    pipe: object, x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> tuple[float, float]:
    if x.empty or x.shape[1] == 0:
        return 0.5, 0.0
    scores = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores)), float(np.std(scores))


def _build_feature_sets(
    schema: DatasetSchema,
    selection: FeatureSelectionResult,
    primary_driver: str,
) -> dict[str, list[str]]:
    raw_signal = (
        selection.selected_numeric
        + selection.selected_binary
        + selection.selected_categorical
    )
    engineered = list(selection.selected_engineered)
    raw_plus_eng = raw_signal + engineered
    all_raw = [
        c
        for c in list(schema.numeric_columns)
        + list(schema.binary_columns)
        + list(schema.categorical)
        if c not in engineered
    ]
    ablation = [c for c in raw_plus_eng if c != primary_driver]
    return {
        "raw_signal": raw_signal,
        "engineered_only": engineered,
        "raw_plus_engineered": raw_plus_eng,
        "all_raw_features": all_raw,
        "ablation_no_primary_driver": ablation,
    }


def _evaluate_one_set(
    name: str,
    cols: list[str],
    work: pd.DataFrame,
    y: pd.Series,
    cat_set: set[str],
    cv: StratifiedKFold,
) -> dict[str, object]:
    valid_cols = [c for c in cols if c in work.columns]
    cat_cols = [c for c in valid_cols if c in cat_set]
    num_cols = [c for c in valid_cols if c not in cat_set]
    x = work.loc[:, valid_cols]
    lr_mean, lr_std = _score_single_model(
        build_logistic_pipeline(num_cols, cat_cols), x, y, cv
    )
    hgb_mean, hgb_std = _score_single_model(
        build_hgb_pipeline(num_cols, cat_cols), x, y, cv
    )
    return {
        "feature_set": name,
        "n_features": len(valid_cols),
        "logreg_roc_auc": round(lr_mean, 4),
        "logreg_std": round(lr_std, 4),
        "hgb_roc_auc": round(hgb_mean, 4),
        "hgb_std": round(hgb_std, 4),
    }


def _lookup_auc(table: pd.DataFrame, name: str, col: str = "hgb_roc_auc") -> float:
    sub = table.loc[table["feature_set"] == name, col]
    return float(sub.iloc[0]) if not sub.empty else 0.5


def evaluate_feature_sets(
    df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    selection: FeatureSelectionResult | None = None,
    n_splits: int = 5,
) -> EvaluationResult:
    require_non_empty(df, "df")
    active = resolve_drivers(df, schema or infer_schema(df))
    if (
        not active.target_column
        or active.target_column not in df.columns
        or selection is None
    ):
        empty_df = pd.DataFrame(
            columns=[
                "feature_set",
                "n_features",
                "logreg_roc_auc",
                "logreg_std",
                "hgb_roc_auc",
                "hgb_std",
            ]
        )
        return EvaluationResult(
            cv_table=empty_df,
            ablation_delta=0.0,
            noise_lift=0.0,
            n_splits=n_splits,
            feature_sets={},
        )
    y = extract_binary_target(df, active.target_column)
    if y.nunique() < 2 or int(y.value_counts().min()) < 2:
        empty_df = pd.DataFrame(
            columns=[
                "feature_set",
                "n_features",
                "logreg_roc_auc",
                "logreg_std",
                "hgb_roc_auc",
                "hgb_std",
            ]
        )
        return EvaluationResult(
            cv_table=empty_df,
            ablation_delta=0.0,
            noise_lift=0.0,
            n_splits=n_splits,
            feature_sets={},
        )
    if int(y.value_counts().min()) < n_splits:
        n_splits = max(2, min(n_splits, int(y.value_counts().min())))
    feature_sets = _build_feature_sets(active, selection, active.primary_driver)
    num_all = list(
        dict.fromkeys(list(active.numeric_columns) + selection.selected_engineered)
    )
    work = _prepare_eval_frame(
        df, num_all, list(active.binary_columns), list(active.categorical)
    )
    cat_set = set(active.categorical)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = [
        _evaluate_one_set(name, cols, work, y, cat_set, cv)
        for name, cols in feature_sets.items()
    ]
    table = pd.DataFrame(rows)
    full_auc = _lookup_auc(table, "raw_plus_engineered")
    ablation_delta = round(
        full_auc - _lookup_auc(table, "ablation_no_primary_driver"), 4
    )
    noise_lift = round(
        _lookup_auc(table, "raw_signal") - _lookup_auc(table, "all_raw_features"), 4
    )
    return EvaluationResult(
        cv_table=table,
        ablation_delta=ablation_delta,
        noise_lift=noise_lift,
        n_splits=n_splits,
        feature_sets=feature_sets,
    )
