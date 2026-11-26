from pathlib import Path

import pandas as pd

from src.preprocess.modeling.workflow import run_preprocess_pipeline
from src.preprocess.outputs.feature_charts import generate_preprocess_plots


def test_generate_preprocess_plots_4_artifacts(
    tmp_path: Path, prep_df: pd.DataFrame
) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    result = run_preprocess_pipeline(df, n_splits=3)
    artifacts = generate_preprocess_plots(result, tmp_path / "assets")
    assert len(artifacts) == 4


def test_generate_preprocess_plots_skips_unidentified(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    result = run_preprocess_pipeline(df, n_splits=2)
    artifacts = generate_preprocess_plots(result, tmp_path / "assets_unid")
    assert len(artifacts) == 4
    assert any(a.skip_reason is not None for a in artifacts)
