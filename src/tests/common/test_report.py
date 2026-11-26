import subprocess
import sys
from pathlib import Path

from src.common.report.assembler import assemble_combined_report


def test_assemble_combined_report_variants(tmp_path: Path) -> None:
    data_frag = tmp_path / "data_frag.md"
    prep_frag = tmp_path / "prep_frag.md"
    data_frag.write_text("## Data Section\n\nData body.\n", encoding="utf-8")
    prep_frag.write_text("## Preprocess Section\n\nPrep body.\n", encoding="utf-8")

    both_path = assemble_combined_report(
        output_path=tmp_path / "both.md",
        data_fragment_path=data_frag,
        preprocess_fragment_path=prep_frag,
    )
    both = both_path.read_text(encoding="utf-8")
    assert "Data Section" in both
    assert "Preprocess Section" in both

    data_only = assemble_combined_report(
        output_path=tmp_path / "d_only.md",
        data_fragment_path=data_frag,
        preprocess_fragment_path=tmp_path / "missing_prep.md",
    ).read_text(encoding="utf-8")
    assert "Data Section" in data_only
    assert "Preprocess Section" not in data_only

    prep_only = assemble_combined_report(
        output_path=tmp_path / "p_only.md",
        data_fragment_path=tmp_path / "missing_data.md",
        preprocess_fragment_path=prep_frag,
    ).read_text(encoding="utf-8")
    assert "Preprocess Section" in prep_only
    assert "Data Section" not in prep_only

    none_md = assemble_combined_report(
        output_path=tmp_path / "none.md",
        data_fragment_path=tmp_path / "missing_d.md",
        preprocess_fragment_path=tmp_path / "missing_p.md",
    ).read_text(encoding="utf-8")
    assert "has not been generated yet" in none_md


def test_common_report_cli(tmp_path: Path) -> None:
    data_frag = tmp_path / "d.md"
    prep_frag = tmp_path / "p.md"
    out_md = tmp_path / "README.md"
    data_frag.write_text("## Data\n\nOK\n", encoding="utf-8")
    prep_frag.write_text("## Prep\n\nOK\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.common.report.assembler",
            "--data-fragment",
            str(data_frag),
            "--preprocess-fragment",
            str(prep_frag),
            "--output",
            str(out_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    content = out_md.read_text(encoding="utf-8")
    assert "## Data" in content
    assert "## Prep" in content
