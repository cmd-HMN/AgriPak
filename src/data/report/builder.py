from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.analysis import AnalysisBundle
from src.data.cleaning import CleaningResult
from src.data.errors import ReportWriteError
from src.data.plotting import PlotArtifact
from src.data.report.formatters import _build_header
from src.data.report.sections import (
    _build_section_1,
    _build_section_2_1,
    _build_section_2_2_and_2_3,
    _build_section_2_4_and_2_5,
    _build_section_2_6_and_2_7,
    _build_section_2_8_and_2_9,
    _build_section_3,
    _build_section_4,
    _build_section_5,
    _build_section_6,
)
from src.data.utils import ensure_directory


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
        "---",
        "",
        *_build_section_5(analysis, plot_map, report_dir),
        "---",
        "",
        *_build_section_6(),
        "---",
        "",
        "## License",
        "",
        "AgriPak is MIT-licensed, as found in the [LICENSE](../../LICENSE) file.",
        "",
    ]
    return "\n".join(sections).rstrip() + "\n"


def generate_report(
    raw_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    analysis: AnalysisBundle,
    plots: Sequence[PlotArtifact],
    input_path: Path,
    report_path: Path,
) -> Path:
    target = Path(report_path)
    ensure_directory(target.parent)
    markdown = build_report_markdown(
        raw_df=raw_df,
        cleaning_result=cleaning_result,
        analysis=analysis,
        plots=plots,
        input_path=input_path,
        report_path=target,
    )
    try:
        target.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(f"Failed to write report to '{target}': {exc}") from exc
    return target
