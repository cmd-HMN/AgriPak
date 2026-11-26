import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from src.common.core.errors import EXIT_SUCCESS
from src.common.core.io import write_text_file
from src.common.report.markdown import render_exit_code_table

DATA_FRAGMENT_PATH = Path("src/data/report_fragment.md")
PREPROCESS_FRAGMENT_PATH = Path("src/preprocess/report_fragment.md")
COMBINED_REPORT_PATH = Path("src/data/README.md")


def relative_asset_prefix(assets_dir: Path, doc_dir: Path) -> str:
    rel = os.path.relpath(Path(assets_dir).resolve(), Path(doc_dir).resolve())
    return Path(rel).as_posix()


def write_fragment(markdown: str, output_path: Path) -> Path:
    return write_text_file(markdown, output_path)


def _read_fragment_or_notice(fragment_path: Path, title: str, cli_command: str) -> str:
    path = Path(fragment_path)
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return "\n".join(
        [
            f"## {title}",
            "",
            f"_Report fragment `{path}` has not been generated yet._",
            "",
            "Run the following command to generate it:",
            "",
            "```bash",
            cli_command,
            "```",
        ]
    )


def _reproducibility_section() -> str:
    return "\n".join(
        [
            "## 6. CLI Exit Codes & Reproducibility",
            "",
            render_exit_code_table(),
            "",
            "```bash",
            "python -m src.data --input data/raw --output-dir src/data/assets",
            "python -m src.preprocess --input data/raw --output data/preprocessed/cleaned.parquet",
            "python -m src.common.report.assembler",
            "```",
            "",
            "## License",
            "",
            "This project is licensed under the MIT License.",
        ]
    )


def assemble_combined_report(
    output_path: Path = COMBINED_REPORT_PATH,
    data_fragment_path: Path = DATA_FRAGMENT_PATH,
    preprocess_fragment_path: Path = PREPROCESS_FRAGMENT_PATH,
) -> Path:
    data_section = _read_fragment_or_notice(
        data_fragment_path,
        "Data Analysis Report",
        "python -m src.data --input data/raw --output-dir src/data/assets",
    )
    preprocess_section = _read_fragment_or_notice(
        preprocess_fragment_path,
        "Preprocessing Report",
        "python -m src.preprocess --input data/raw --output data/preprocessed/cleaned.parquet",
    )
    combined = (
        "\n\n".join([data_section, preprocess_section, _reproducibility_section()])
        + "\n"
    )
    return write_text_file(combined, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.common.report")
    parser.add_argument("--data-fragment", type=Path, default=DATA_FRAGMENT_PATH)
    parser.add_argument(
        "--preprocess-fragment", type=Path, default=PREPROCESS_FRAGMENT_PATH
    )
    parser.add_argument("--output", type=Path, default=COMBINED_REPORT_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    assemble_combined_report(
        output_path=args.output,
        data_fragment_path=args.data_fragment,
        preprocess_fragment_path=args.preprocess_fragment,
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
