from pathlib import Path

import pandas as pd
import pytest

from src.data.analysis import run_full_analysis
from src.data.cleaning import clean_dataset
from src.data.errors import ReportWriteError
from src.data.plotting import (
    generate_all_plots,
    plot_binary_flag_counts,
    plot_categorical_counts,
    plot_continuous_binary_interactions,
    plot_cramers_v_associations,
    plot_decile_risk_curves,
    plot_engineered_features,
    plot_kruskal_wallis_boxplots,
    plot_numeric_by_target,
    plot_numeric_distributions,
    plot_pairwise_risk_heatmaps,
    plot_risk_rate_by_category,
    plot_spearman_correlation,
    plot_subgroup_risk_surfaces,
    plot_target_distribution,
    plot_top_combinations,
)
from src.data.schema import infer_schema


def test_all_15_plot_functions_and_generate_all_plots(
    synthetic_clean_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    clean_res = clean_dataset(synthetic_clean_df, target_column="target_bin")
    bundle = run_full_analysis(clean_res.data, clean_res.schema, min_n=5, n_splits=3)
    assets_dir = tmp_path / "assets"

    individual = [
        plot_target_distribution(clean_res.data, clean_res.schema, assets_dir),
        plot_numeric_distributions(clean_res.data, clean_res.schema, assets_dir),
        plot_categorical_counts(clean_res.data, clean_res.schema, assets_dir),
        plot_binary_flag_counts(clean_res.data, clean_res.schema, assets_dir),
        plot_risk_rate_by_category(clean_res.data, clean_res.schema, bundle.base_rate, assets_dir),
        plot_numeric_by_target(clean_res.data, clean_res.schema, assets_dir),
        plot_spearman_correlation(bundle.spearman_matrix, assets_dir),
        plot_decile_risk_curves(bundle.decile_curves, bundle.base_rate, clean_res.schema.target_column, assets_dir),
        plot_pairwise_risk_heatmaps(clean_res.data, clean_res.schema, bundle.base_rate, bundle.min_n, assets_dir),
        plot_top_combinations(bundle.combinations, bundle.base_rate, clean_res.schema.target_column, assets_dir),
        plot_continuous_binary_interactions(clean_res.data, clean_res.schema, bundle.base_rate, assets_dir),
        plot_subgroup_risk_surfaces(bundle.subgroup_surfaces, clean_res.schema, bundle.base_rate, assets_dir),
        plot_kruskal_wallis_boxplots(clean_res.data, clean_res.schema, bundle.kruskal_wallis, assets_dir),
        plot_cramers_v_associations(bundle.target_cramers_v, bundle.pairwise_cramers_v, clean_res.schema.target_column, assets_dir),
        plot_engineered_features(bundle.feature_engineering, clean_res.schema, bundle.base_rate, assets_dir),
    ]
    assert len(individual) == 15
    for artifact in individual:
        if artifact.generated:
            assert artifact.path.exists()
            raw_bytes = artifact.path.read_bytes()
            assert len(raw_bytes) > 0
            assert raw_bytes.startswith(b"\x89PNG")

    batch = generate_all_plots(clean_res.data, bundle, tmp_path / "assets_batch")
    assert len(batch) == 15
    assert any(item.generated for item in batch)


def test_plots_gracefully_skip_on_unidentified_schema(
    synthetic_unidentifiable_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    schema = infer_schema(synthetic_unidentifiable_df)
    bundle = run_full_analysis(synthetic_unidentifiable_df, schema, min_n=5, n_splits=3)
    artifacts = generate_all_plots(synthetic_unidentifiable_df, bundle, tmp_path / "assets_unid")
    assert len(artifacts) == 15
    assert all(not item.generated for item in artifacts)


def test_plot_raises_report_write_error_on_unwritable_path(
    synthetic_clean_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    schema = infer_schema(synthetic_clean_df, target_column="target_bin")
    blocker = tmp_path / "file_blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ReportWriteError):
        plot_target_distribution(synthetic_clean_df, schema, blocker)
