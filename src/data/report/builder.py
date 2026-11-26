from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common.report.assembler import (
    COMBINED_REPORT_PATH,
    DATA_FRAGMENT_PATH,
    PREPROCESS_FRAGMENT_PATH,
    assemble_combined_report,
    write_fragment,
)
from src.data.analysis import AnalysisBundle
from src.data.cleaning import CleaningResult
from src.data.plotting import PlotArtifact
from src.data.report.formatters import _build_header
from src.data.report.sections_eda import (
    _build_section_2_1,
    _build_section_2_2_and_2_3,
    _build_section_2_4_and_2_5,
    _build_section_2_6_and_2_7,
    _build_section_2_8_and_2_9,
)
from src.data.report.sections_summary import (
    _build_section_1,
    _build_section_3,
    _build_section_4,
)


def build_report_markdown(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    analysis: AnalysisBundle,
    plots: Sequence[PlotArtifact],
    input_path: Path,
    report_path: Path,
) -> str:
    report_dir = Path(report_path).parent
    plot_map = {plot.plot_id: plot for plot in plots}
    sections: list[str] = [
        *_build_header(raw_df, cleaning_result, analysis, plots, Path(input_path)),
        *_build_section_1(raw_df, analysis, plot_map, report_dir, Path(input_path)),
        "---",
        "",
        "## 2. Data Quality & Exploratory Statistical Audit",
        "",
        *_build_section_2_1(analysis),
        *_build_section_2_2_and_2_3(analysis, plot_map, report_dir),
        *_build_section_2_4_and_2_5(analysis, plot_map, report_dir),
        *_build_section_2_6_and_2_7(analysis, plot_map, report_dir),
        *_build_section_2_8_and_2_9(analysis, plot_map, report_dir),
        "---",
        "",
        *_build_section_3(cleaning_result),
        "---",
        "",
        *_build_section_4(analysis),
    ]
    return "\n".join(sections).rstrip() + "\n"


def build_data_fragment(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    analysis: AnalysisBundle,
    plots: Sequence[PlotArtifact],
    input_path: Path,
    fragment_path: Path = DATA_FRAGMENT_PATH,
) -> Path:
    markdown = build_report_markdown(
        raw_df, cleaning_result, analysis, plots, input_path, fragment_path
    )
    return write_fragment(markdown, fragment_path)


def generate_report(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    analysis: AnalysisBundle,
    plots: Sequence[PlotArtifact],
    input_path: Path,
    report_path: Path = COMBINED_REPORT_PATH,
    fragment_path: Path = DATA_FRAGMENT_PATH,
    preprocess_fragment_path: Path = PREPROCESS_FRAGMENT_PATH,
) -> Path:
    build_data_fragment(
        raw_df, cleaning_result, analysis, plots, input_path, fragment_path
    )
    return assemble_combined_report(
        data_fragment_path=fragment_path,
        preprocess_fragment_path=preprocess_fragment_path,
        output_path=report_path,
    )
