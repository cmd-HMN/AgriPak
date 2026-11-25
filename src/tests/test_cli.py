from pathlib import Path
import subprocess
import sys

import pytest

from src.data import cli
from src.data.errors import (
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    AnalysisError,
)


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["run", "--help"],
        ["audit", "--help"],
        ["clean", "--help"],
        ["analyze", "--help"],
        ["features", "--help"],
        ["plot", "--help"],
        ["report", "--help"],
    ],
)
def test_cli_help_flags(argv: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.data", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()


@pytest.mark.parametrize(
    "subcommand",
    ["audit", "clean", "analyze", "features", "plot", "report", "run"],
)
def test_cli_all_subcommands_on_synthetic_csv(
    subcommand: str,
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / f"out_{subcommand}"
    assets_dir = out_dir / "assets"
    code = cli.main(
        [
            subcommand,
            "--input",
            str(synthetic_csv_path),
            "--output",
            str(out_dir),
            "--assets",
            str(assets_dir),
            "--min-n",
            "5",
            "--n-splits",
            "3",
        ]
    )
    assert code == EXIT_SUCCESS


def test_cli_exit_code_2_invalid_arguments(synthetic_csv_path: Path, tmp_path: Path) -> None:
    assert cli.main(["--unknown-flag"]) == 2
    assert (
        cli.main(
            [
                "audit",
                "--input",
                str(synthetic_csv_path),
                "--output",
                str(tmp_path),
                "--n-splits",
                "1",
            ]
        )
        == 2
    )
    assert (
        cli.main(
            [
                "audit",
                "--input",
                str(synthetic_csv_path),
                "--output",
                str(tmp_path),
                "--leak-threshold",
                "1.5",
            ]
        )
        == 2
    )


def test_cli_exit_code_3_missing_input(tmp_path: Path) -> None:
    code = cli.main(["audit", "--input", str(tmp_path / "missing.csv"), "--output", str(tmp_path)])
    assert code == 3


def test_cli_exit_code_4_empty_data(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_bytes(b"")
    code = cli.main(["audit", "--input", str(empty_csv), "--output", str(tmp_path)])
    assert code == 4


def test_cli_exit_code_5_malformed_data(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("a,b\n1,2\n", encoding="utf-8")
    code = cli.main(["audit", "--input", str(bad_file), "--output", str(tmp_path)])
    assert code == 5


def test_cli_exit_code_6_missing_target_column(
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    code = cli.main(
        [
            "audit",
            "--input",
            str(synthetic_csv_path),
            "--output",
            str(tmp_path),
            "--target",
            "does_not_exist",
        ]
    )
    assert code == 6


def test_cli_exit_code_7_analysis_error(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise AnalysisError("synthetic analysis failure")

    monkeypatch.setattr(cli, "run_full_analysis", _boom)
    code = cli.main(
        [
            "analyze",
            "--input",
            str(synthetic_csv_path),
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 7


def test_cli_exit_code_8_report_write_error(
    synthetic_csv_path: Path,
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code = cli.main(
        [
            "audit",
            "--input",
            str(synthetic_csv_path),
            "--output",
            str(blocker),
        ]
    )
    assert code == 8


def test_cli_exit_code_99_and_debug_traceback(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_csv_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _crash(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "audit_data_quality", _crash)

    code_no_debug = cli.main(
        ["audit", "--input", str(synthetic_csv_path), "--output", str(tmp_path)]
    )
    err_no_debug = capsys.readouterr().err
    assert code_no_debug == EXIT_UNEXPECTED
    assert "Traceback" not in err_no_debug

    code_debug = cli.main(
        ["audit", "--input", str(synthetic_csv_path), "--output", str(tmp_path), "--debug"]
    )
    err_debug = capsys.readouterr().err
    assert code_debug == EXIT_UNEXPECTED
    assert "Traceback" in err_debug
