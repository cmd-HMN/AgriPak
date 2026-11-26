from pathlib import Path

import pandas as pd

from src.common.dataset.columns import infer_schema, resolve_drivers
from src.data.analysis import run_full_analysis
from src.data.plotting import generate_all_plots


def test_generate_all_plots_produces_14_artifacts(
    tmp_path: Path, eda_df: pd.DataFrame
) -> None:
    schema = resolve_drivers(eda_df, infer_schema(eda_df))
    result = run_full_analysis(eda_df, schema)
    artifacts = generate_all_plots(eda_df, result, tmp_path / "assets")
    assert len(artifacts) == 14


def test_generate_all_plots_skips_when_unidentified(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    schema = resolve_drivers(df, infer_schema(df))
    result = run_full_analysis(df, schema)
    artifacts = generate_all_plots(df, result, tmp_path / "assets_unid")
    assert len(artifacts) == 14
    assert any(a.skip_reason is not None for a in artifacts)
