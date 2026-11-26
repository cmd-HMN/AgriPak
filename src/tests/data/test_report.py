import re
from pathlib import Path

import pandas as pd

from src.data.analysis import run_full_analysis
from src.data.cleaning import clean_dataset
from src.data.plotting import generate_all_plots
from src.data.report import build_data_fragment


def test_write_data_report_fragment_and_links(
    tmp_path: Path, eda_df: pd.DataFrame
) -> None:
    res = clean_dataset(eda_df)
    result = run_full_analysis(res.data, res.schema)
    assets_dir = tmp_path / "assets"
    artifacts = generate_all_plots(res.data, result, assets_dir)
    frag_path = tmp_path / "report_fragment.md"
    build_data_fragment(
        raw_df=eda_df,
        cleaning_result=res,
        analysis=result,
        plots=artifacts,
        input_path=tmp_path / "input.csv",
        fragment_path=frag_path,
    )
    assert frag_path.is_file()
    text = frag_path.read_text(encoding="utf-8")
    for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if match.startswith(("http://", "https://")):
            continue
        assert (frag_path.parent / match).exists()
