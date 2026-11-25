from pathlib import Path
import re

import pandas as pd
import pytest

from src.data.analysis import run_full_analysis
from src.data.cleaning import clean_dataset
from src.data.errors import ReportWriteError
from src.data.plotting import generate_all_plots
from src.data.report import build_report_markdown, generate_report
from src.data.utils import UNIDENTIFIED


def test_build_and_generate_report_rich_schema(
    synthetic_clean_df: pd.DataFrame,
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    clean_res = clean_dataset(synthetic_clean_df, target_column="target_bin")
    bundle = run_full_analysis(clean_res.data, clean_res.schema, min_n=5, n_splits=3)
    assets_dir = tmp_path / "assets"
    plots = generate_all_plots(clean_res.data, bundle, assets_dir)
    report_path = tmp_path / "README.md"

    md = build_report_markdown(
        raw_df=synthetic_clean_df,
        cleaning_result=clean_res,
        analysis=bundle,
        plots=plots,
        input_path=synthetic_csv_path,
        report_path=report_path,
    )
    assert "## 1. Dataset Overview & Target Balance" in md
    assert "## 2. Data Quality & Exploratory Statistical Audit" in md
    assert "## 3. Data Cleaning Audit" in md
    assert "## 4. Leakage Sanity Check" in md
    assert "## 5. Feature Engineering & Model Validation" in md
    assert "## 6. CLI Exit Codes & Reproducibility" in md

    written = generate_report(
        raw_df=synthetic_clean_df,
        cleaning_result=clean_res,
        analysis=bundle,
        plots=plots,
        input_path=synthetic_csv_path,
        report_path=report_path,
    )
    assert written.exists()
    content = written.read_text(encoding="utf-8")
    image_links = [
        link
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
        if not link.startswith(("http://", "https://"))
    ]
    assert len(image_links) > 0
    for rel_link in image_links:
        assert (report_path.parent / rel_link).exists()


def test_report_renders_unidentified_fallback(
    synthetic_unidentifiable_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    clean_res = clean_dataset(synthetic_unidentifiable_df)
    bundle = run_full_analysis(clean_res.data, clean_res.schema, min_n=5, n_splits=3)
    plots = generate_all_plots(clean_res.data, bundle, tmp_path / "assets_unid")
    report_path = tmp_path / "unid_README.md"

    md = build_report_markdown(
        raw_df=synthetic_unidentifiable_df,
        cleaning_result=clean_res,
        analysis=bundle,
        plots=plots,
        input_path=tmp_path / "unid.csv",
        report_path=report_path,
    )
    assert UNIDENTIFIED in md


def test_generate_report_raises_on_unwritable_path(
    synthetic_clean_df: pd.DataFrame,
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    clean_res = clean_dataset(synthetic_clean_df, target_column="target_bin")
    bundle = run_full_analysis(clean_res.data, clean_res.schema, min_n=5, n_splits=3)
    plots = generate_all_plots(clean_res.data, bundle, tmp_path / "assets")
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(ReportWriteError):
        generate_report(
            raw_df=synthetic_clean_df,
            cleaning_result=clean_res,
            analysis=bundle,
            plots=plots,
            input_path=synthetic_csv_path,
            report_path=blocker / "README.md",
        )
