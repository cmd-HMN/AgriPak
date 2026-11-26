import re
from pathlib import Path

import pandas as pd

from src.preprocess.modeling.workflow import run_preprocess_pipeline
from src.preprocess.outputs.feature_charts import generate_preprocess_plots
from src.preprocess.outputs.reporting import generate_preprocess_reports


def test_preprocess_report_single_source_of_truth(
    tmp_path: Path, prep_df: pd.DataFrame
) -> None:
    df = pd.concat([prep_df, prep_df], ignore_index=True)
    result = run_preprocess_pipeline(df, n_splits=3)
    prep_dir = tmp_path / "preprocess"
    assets_dir = prep_dir / "assets"
    artifacts = generate_preprocess_plots(result, assets_dir)
    readme_path, frag_path = generate_preprocess_reports(
        bundle=result,
        plots=artifacts,
        output_dir=prep_dir,
        asset_prefix="assets",
        combined_report_path=None,
    )
    frag_text = frag_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "Preprocessing & Feature Engineering" in readme_text
    assert len(frag_text) > 0
    for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme_text):
        assert (readme_path.parent / match).exists()
