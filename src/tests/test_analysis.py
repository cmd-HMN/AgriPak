import numpy as np
import pandas as pd

from src.data.analysis import (
    analyze_feature_combinations,
    audit_data_quality,
    compute_categorical_summaries,
    compute_cramers_v,
    compute_cramers_v_tables,
    compute_decile_target_curves,
    compute_numeric_shape_metrics,
    compute_ranked_target_correlations,
    compute_spearman_matrix,
    compute_subgroup_risk_surfaces,
    compute_target_rates_by_group,
    engineer_features,
    infer_risk_floor_threshold,
    run_full_analysis,
    run_kruskal_wallis_grid,
    run_leakage_check,
    run_mann_whitney_tests,
    validate_models_cv,
)
from src.data.cleaning import clean_dataset
from src.data.schema import infer_schema


def test_audit_and_shape_and_categorical_summaries(synthetic_clean_df: pd.DataFrame) -> None:
    qa = audit_data_quality(synthetic_clean_df, raw_text_df=synthetic_clean_df.astype(str))
    assert qa.duplicate_count == 0
    assert qa.memory_mb > 0.0
    assert len(qa.column_summary) == len(synthetic_clean_df.columns)
    assert not qa.text_hygiene.empty

    shape = compute_numeric_shape_metrics(synthetic_clean_df, ["num_a", "num_b"])
    assert set(shape["column"]) == {"num_a", "num_b"}
    assert "skewness" in shape.columns

    cat_counts = compute_categorical_summaries(synthetic_clean_df, ["cat_1", "cat_2"])
    assert set(cat_counts.keys()) == {"cat_1", "cat_2"}
    assert not cat_counts["cat_1"].empty

    rates = compute_target_rates_by_group(synthetic_clean_df, ["cat_1", "flag_x"], "target_bin")
    assert set(rates.keys()) == {"cat_1", "flag_x"}
    assert "target_rate" in rates["cat_1"].columns


def test_mann_whitney_spearman_deciles_cramers_kruskal(synthetic_clean_df: pd.DataFrame) -> None:
    mw = run_mann_whitney_tests(synthetic_clean_df, ["num_a", "num_b"], "target_bin")
    assert len(mw) == 2
    assert "p_adj_fdr" in mw.columns

    spearman = compute_spearman_matrix(synthetic_clean_df, ["num_a", "num_b", "num_c", "target_bin"])
    assert spearman.shape == (4, 4)

    ranked = compute_ranked_target_correlations(spearman, "target_bin")
    assert not ranked.empty
    assert "abs_spearman_r" in ranked.columns

    deciles = compute_decile_target_curves(synthetic_clean_df, ["num_a", "num_b"], "target_bin", n_bins=5)
    assert "num_a" in deciles
    assert not deciles["num_a"].empty

    v, p = compute_cramers_v(synthetic_clean_df["cat_1"], synthetic_clean_df["target_bin"])
    assert 0.0 <= v <= 1.0
    assert 0.0 <= p <= 1.0

    target_v, pair_v = compute_cramers_v_tables(
        synthetic_clean_df,
        ["cat_1", "cat_2", "flag_x"],
        "target_bin",
    )
    assert not target_v.empty
    assert pair_v.shape == (3, 3)

    kw = run_kruskal_wallis_grid(synthetic_clean_df, ["num_a", "num_b"], ["cat_1", "cat_2"])
    assert len(kw) == 4
    assert "p_adj_fdr" in kw.columns


def test_combinations_and_subgroup_surfaces(synthetic_clean_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_clean_df, target_column="target_bin")
    combos = analyze_feature_combinations(synthetic_clean_df, schema, min_n=5, ks=(2,), top_n=5)
    assert combos.total_combos > 0
    assert not combos.top_risky.empty

    surfaces = compute_subgroup_risk_surfaces(
        synthetic_clean_df,
        driver_x=schema.primary_driver,
        driver_y=schema.secondary_driver,
        target_col="target_bin",
        subgroup_col="cat_1",
        min_n=3,
    )
    assert isinstance(surfaces, dict)


def test_leakage_check_flags_synthetic_leaked_column(synthetic_clean_df: pd.DataFrame) -> None:
    df = synthetic_clean_df.copy()
    df["leaked_col"] = df["target_bin"].astype(float) + np.random.default_rng(1).normal(0, 0.001, len(df))
    schema = infer_schema(df, target_column="target_bin")
    leak_df, flagged = run_leakage_check(df, schema, leak_threshold=0.90, auc_threshold=0.95, n_splits=3)
    assert not leak_df.empty
    assert "leaked_col" in flagged


def test_threshold_feature_engineering_and_cv(synthetic_clean_df: pd.DataFrame) -> None:
    schema = infer_schema(synthetic_clean_df, target_column="target_bin")
    threshold, direction = infer_risk_floor_threshold(
        synthetic_clean_df,
        schema.primary_driver,
        "target_bin",
        n_bins=5,
    )
    assert direction in {"below", "above", "none"}
    assert threshold is None or isinstance(threshold, float)

    fe_result = engineer_features(synthetic_clean_df, schema, threshold=threshold)
    assert len(fe_result.engineered_columns) > 0
    assert not fe_result.summary_stats.empty

    cv_table = validate_models_cv(synthetic_clean_df, schema, fe_result, n_splits=3, random_state=42)
    assert len(cv_table) == 3
    assert "mean_auc" in cv_table.columns


def test_run_full_analysis_rich_and_unidentified(
    synthetic_clean_df: pd.DataFrame,
    synthetic_unidentifiable_df: pd.DataFrame,
) -> None:
    clean_res = clean_dataset(synthetic_clean_df, target_column="target_bin")
    bundle = run_full_analysis(clean_res.data, clean_res.schema, min_n=5, n_splits=3)
    assert bundle.row_count == len(clean_res.data)
    assert bundle.base_rate is not None
    assert not bundle.cv_auc_table.empty

    unid_schema = infer_schema(synthetic_unidentifiable_df)
    unid_bundle = run_full_analysis(synthetic_unidentifiable_df, unid_schema, min_n=5, n_splits=3)
    assert unid_bundle.base_rate is None
    assert unid_bundle.cv_auc_table.empty
