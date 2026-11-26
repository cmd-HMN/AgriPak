import pandas as pd

from src.common.dataset.columns import infer_schema, resolve_drivers
from src.data.analysis import run_full_analysis


def test_run_full_analysis_and_no_fe_fields(eda_df: pd.DataFrame) -> None:
    schema = resolve_drivers(eda_df, infer_schema(eda_df))
    result = run_full_analysis(eda_df, schema)
    assert result.schema.target_column == "high_risk"
    assert not result.quality_audit.column_summary.empty
    assert not result.numeric_shape.empty
    assert not hasattr(result, "engineered_features")
    assert not hasattr(result, "selected_features")
