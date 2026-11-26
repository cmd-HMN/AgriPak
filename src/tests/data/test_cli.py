import subprocess
import sys
from pathlib import Path


def _run_data_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.data", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_data_cli_commands(tmp_path: Path, eda_csv_path: Path) -> None:
    cleaned_out = tmp_path / "cleaned.csv"
    assert _run_data_cli(["audit", "--input", str(eda_csv_path)]).returncode == 0
    assert (
        _run_data_cli(
            ["clean", "--input", str(eda_csv_path), "--output", str(cleaned_out)]
        ).returncode
        == 0
    )
    assert _run_data_cli(["analyze", "--input", str(eda_csv_path)]).returncode == 0
    assert (
        _run_data_cli(
            ["plot", "--input", str(eda_csv_path), "--assets", str(tmp_path / "plots")]
        ).returncode
        == 0
    )
    assert (
        _run_data_cli(
            [
                "report",
                "--input",
                str(eda_csv_path),
                "--report",
                str(tmp_path / "rep" / "README.md"),
                "--fragment",
                str(tmp_path / "rep" / "fragment.md"),
                "--assets",
                str(tmp_path / "rep_assets"),
            ]
        ).returncode
        == 0
    )
    assert (
        _run_data_cli(
            [
                "run",
                "--input",
                str(eda_csv_path),
                "--output",
                str(tmp_path / "all_out" / "cleaned.csv"),
                "--report",
                str(tmp_path / "all_out" / "README.md"),
                "--fragment",
                str(tmp_path / "all_out" / "fragment.md"),
                "--assets",
                str(tmp_path / "all_assets"),
            ]
        ).returncode
        == 0
    )


def test_data_cli_features_fails_with_code_2() -> None:
    proc = _run_data_cli(["features"])
    assert proc.returncode == 2


def test_data_cli_exit_codes_and_debug(tmp_path: Path) -> None:
    assert (
        _run_data_cli(
            [
                "clean",
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(tmp_path / "o.csv"),
            ]
        ).returncode
        == 3
    )
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert (
        _run_data_cli(
            ["clean", "--input", str(empty), "--output", str(tmp_path / "o2.csv")]
        ).returncode
        == 4
    )
    bad = tmp_path / "bad.xyz"
    bad.write_text("bad", encoding="utf-8")
    assert (
        _run_data_cli(
            ["clean", "--input", str(bad), "--output", str(tmp_path / "o3.csv")]
        ).returncode
        == 5
    )
    proc_dbg = _run_data_cli(
        [
            "--debug",
            "clean",
            "--input",
            str(tmp_path / "missing.csv"),
            "--output",
            str(tmp_path / "o4.csv"),
        ]
    )
    assert proc_dbg.returncode == 3
