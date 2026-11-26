from pathlib import Path

from src.common.core.io import ensure_directory, write_text_file
from src.common.core.validation import UNIDENTIFIED
from src.common.report.assembler import (
    COMBINED_REPORT_PATH,
    assemble_combined_report,
    write_fragment,
)
from src.common.report.markdown import (
    append_markdown_table,
    render_exit_code_table,
    render_plot_markdown,
)
from src.common.report.plotting import PlotArtifact
from src.preprocess.modeling.workflow import PreprocessBundle


def _find_plot(plots: list[PlotArtifact], filename: str) -> PlotArtifact | None:
    for art in plots:
        if art.path is not None and art.path.name == filename:
            return art
    return None


def _render_overview_section(
    bundle: PreprocessBundle, plots: list[PlotArtifact], asset_prefix: str
) -> list[str]:
    eng = bundle.engineered
    driver = eng.primary_driver or UNIDENTIFIED
    lines = [
        "## Preprocessing & Feature Engineering",
        "",
        f"- **Rows / Columns:** `{len(bundle.raw_df):,}` rows, `{bundle.raw_df.shape[1]}` raw columns -> `{bundle.processed_df.shape[1]}` processed columns",
        f"- **Primary Driver:** `{driver}` (inferred risk floor threshold: `{eng.risk_floor_threshold:.2f}`)",
        f"- **Binary Signal Columns ({len(eng.binary_signal_cols)}):** `{', '.join(eng.binary_signal_cols) or UNIDENTIFIED}`",
        f"- **Engineered Features ({len(eng.engineered_cols)}):** `{', '.join(eng.engineered_cols)}`",
        "",
        "### Engineered Feature Definitions",
        "",
        f"1. `{eng.signal_sum_col}`: Row-wise sum of active binary risk flags (`{len(eng.binary_signal_cols)}` columns), capturing cumulative risk.",
        f"2. `{eng.deficit_col}`: `max(0, {eng.risk_floor_threshold:.2f} - {driver})`, isolating shortfall below the positive-class median.",
        f"3. `{eng.deficit_x_zero_col}`: Interaction `{eng.deficit_col} * I({eng.signal_sum_col} == 0)`, capturing silent-risk cases.",
        f"4. `{eng.driver_x_sum_col}`: Interaction `{driver} * {eng.signal_sum_col}`, scaling driver magnitude by active flag count.",
        "",
    ]
    for fname in (
        "engineered_features.png",
        "signal_sum_risk.png",
        "driver_by_signal_curves.png",
    ):
        art = _find_plot(plots, fname)
        if art is not None:
            lines.extend(
                render_plot_markdown(
                    {fname: art},
                    plot_id=fname,
                    fallback_title=fname,
                    asset_prefix=asset_prefix,
                )
            )
    return lines


def _render_selection_section(bundle: PreprocessBundle) -> list[str]:
    sel = bundle.selection
    lines = [
        "## Feature Selection & Transformation Summary",
        "",
        f"- **Selected Features ({len(sel.selected_features)}):** `{', '.join(sel.selected_features) or UNIDENTIFIED}`",
        f"- **Dropped Noise Features ({len(sel.dropped_features)}):** `{', '.join(sel.dropped_features) or 'None'}`",
        f"- **Encoding (`{bundle.encoding.method}`):** `{len(bundle.encoding.source_columns)}` categorical -> `{len(bundle.encoding.encoded_columns)}` encoded columns",
        f"- **Scaling (`{bundle.scaling.method}`):** `{len(bundle.scaling.scaled_columns)}` numeric columns (`{len(bundle.scaling.zero_variance_columns)}` zero-variance)",
        "",
    ]
    append_markdown_table(lines, "Selection Manifest", sel.manifest)
    return lines


def _render_evaluation_section(
    bundle: PreprocessBundle, plots: list[PlotArtifact], asset_prefix: str
) -> list[str]:
    ev = bundle.evaluation
    lines = [
        f"## {ev.n_splits}-Fold Stratified CV & Ablation",
        "",
        f"- **Primary Driver Ablation Delta (HGB):** `{ev.ablation_delta:+.4f}` ROC-AUC",
        f"- **Noise Removal Lift (HGB):** `{ev.noise_lift:+.4f}` ROC-AUC",
        "",
    ]
    append_markdown_table(lines, "Cross-Validation Results", ev.cv_table)
    art = _find_plot(plots, "cv_model_comparison.png")
    if art is not None:
        lines.extend(
            render_plot_markdown(
                {"cv_model_comparison.png": art},
                plot_id="cv_model_comparison.png",
                fallback_title="cv_model_comparison.png",
                asset_prefix=asset_prefix,
            )
        )
    return lines


def render_preprocess_sections(
    bundle: PreprocessBundle, plots: list[PlotArtifact], asset_prefix: str
) -> str:
    sections: list[str] = []
    sections.extend(_render_overview_section(bundle, plots, asset_prefix))
    sections.extend(_render_selection_section(bundle))
    sections.extend(_render_evaluation_section(bundle, plots, asset_prefix))
    return "\n".join(sections).strip() + "\n"


def build_preprocess_fragment(
    bundle: PreprocessBundle,
    plots: list[PlotArtifact],
    asset_prefix: str,
    readme_link: str = "../preprocess/README.md",
) -> str:
    body = render_preprocess_sections(bundle, plots, asset_prefix)
    return f"{body}\n---\n\nFull preprocessing documentation: [`src/preprocess/README.md`]({readme_link})\n"


def _readme_header(combined_report_link: str, root_readme_link: str) -> list[str]:
    return [
        '<div align="center">',
        "",
        "# AgriPak Preprocessing & Feature Engineering",
        "",
        f"[Combined Report]({combined_report_link}) · [Repository Root]({root_readme_link})",
        "",
        "</div>",
        "",
        "## Pipeline Order",
        "",
        "1. `infer_schema` (dynamic column/target/driver detection)",
        "2. `engineer_features` (binary signal sum, driver deficit, interactions)",
        "3. `select_features` (Spearman & Cramer's V filtering)",
        "4. `encode_categorical_columns` & `scale_numeric_columns`",
        "5. `evaluate_feature_sets` (5-fold StratifiedKFold ROC-AUC + ablation)",
        "",
    ]


def _readme_footer() -> list[str]:
    lines = [
        "## Run",
        "",
        "```bash",
        "uv run python -m src.preprocess run --input data/raw --output src/preprocess --assets src/preprocess/assets",
        "uv run python -m src.preprocess evaluate --input data/raw",
        "uv run python -m src.preprocess plot --input data/raw --assets src/preprocess/assets",
        "```",
        "",
    ]
    lines.append(render_exit_code_table())
    lines.extend(
        [
            "",
            "## License",
            "",
            "MIT",
            "",
        ]
    )
    return lines


def build_preprocess_readme(
    bundle: PreprocessBundle,
    plots: list[PlotArtifact],
    asset_prefix: str,
    combined_report_link: str = "../data/README.md",
    root_readme_link: str = "../../README.md",
) -> str:
    header = "\n".join(_readme_header(combined_report_link, root_readme_link))
    body = render_preprocess_sections(bundle, plots, asset_prefix)
    footer = "\n".join(_readme_footer())
    return f"{header}{body}\n{footer}"


def generate_preprocess_reports(
    bundle: PreprocessBundle,
    plots: list[PlotArtifact],
    output_dir: Path,
    asset_prefix: str,
    combined_report_path: Path | None = COMBINED_REPORT_PATH,
) -> tuple[Path, Path]:
    ensure_directory(output_dir)
    readme_path = output_dir / "README.md"
    fragment_path = output_dir / "report_fragment.md"
    readme_text = build_preprocess_readme(bundle, plots, asset_prefix)
    fragment_text = build_preprocess_fragment(bundle, plots, asset_prefix)
    write_text_file(readme_text, readme_path)
    write_fragment(fragment_text, fragment_path)
    if combined_report_path is not None:
        assemble_combined_report(
            output_path=combined_report_path, preprocess_fragment_path=fragment_path
        )
    return readme_path, fragment_path
