import subprocess
import sys
from pathlib import Path


def _run_prep_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.preprocess", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_preprocess_cli_commands(tmp_path: Path, prep_csv_path: Path) -> None:
    feat_out = tmp_path / "feat_out"
    assert (
        _run_prep_cli(
            ["transform", "--input", str(prep_csv_path), "--output", str(feat_out)]
        ).returncode
        == 0
    )
    assert _run_prep_cli(["encode", "--input", str(prep_csv_path)]).returncode == 0
    assert _run_prep_cli(["scale", "--input", str(prep_csv_path)]).returncode == 0
    assert _run_prep_cli(["select", "--input", str(prep_csv_path)]).returncode == 0
    assert _run_prep_cli(["evaluate", "--input", str(prep_csv_path)]).returncode == 0
    assert (
        _run_prep_cli(
            ["plot", "--input", str(prep_csv_path), "--assets", str(tmp_path / "plots")]
        ).returncode
        == 0
    )
    assert (
        _run_prep_cli(
            [
                "report",
                "--input",
                str(prep_csv_path),
                "--output",
                str(tmp_path / "rep"),
                "--assets",
                str(tmp_path / "rep_assets"),
            ]
        ).returncode
        == 0
    )
    assert (
        _run_prep_cli(
            [
                "run",
                "--input",
                str(prep_csv_path),
                "--output",
                str(tmp_path / "all_out"),
                "--assets",
                str(tmp_path / "all_assets"),
            ]
        ).returncode
        == 0
    )


def test_preprocess_cli_exit_codes_and_debug(
    tmp_path: Path, prep_csv_path: Path
) -> None:
    assert _run_prep_cli(["unknown_cmd"]).returncode == 2
    assert (
        _run_prep_cli(
            [
                "transform",
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(tmp_path / "o"),
            ]
        ).returncode
        == 3
    )
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert (
        _run_prep_cli(
            ["transform", "--input", str(empty), "--output", str(tmp_path / "o2")]
        ).returncode
        == 4
    )
    assert (
        _run_prep_cli(
            [
                "encode",
                "--input",
                str(prep_csv_path),
                "--output",
                str(tmp_path / "o3"),
                "--encode-method",
                "bad",
            ]
        ).returncode
        == 9
    )
    assert (
        _run_prep_cli(
            [
                "--debug",
                "transform",
                "--input",
                str(tmp_path / "missing.csv"),
                "--output",
                str(tmp_path / "o4"),
            ]
        ).returncode
        == 3
    )
