from src.data.errors import (
    EXIT_CODE_TABLE,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    AnalysisError,
    DataPipelineError,
    EmptyDataError,
    InputNotFoundError,
    InvalidArgumentError,
    MalformedDataError,
    MissingColumnError,
    ReportWriteError,
    format_cli_error,
)


def test_error_hierarchy_and_exit_codes() -> None:
    expected = [
        (InvalidArgumentError, 2),
        (InputNotFoundError, 3),
        (EmptyDataError, 4),
        (MalformedDataError, 5),
        (MissingColumnError, 6),
        (AnalysisError, 7),
        (ReportWriteError, 8),
    ]
    seen_codes: list[int] = []
    for err_cls, code in expected:
        assert issubclass(err_cls, DataPipelineError)
        err = err_cls("  bad   input \n here ")
        assert err.exit_code == code
        assert err.message == "bad input here"
        seen_codes.append(code)
    assert len(set(seen_codes)) == len(expected)
    assert EXIT_SUCCESS == 0
    assert EXIT_UNEXPECTED == 99
    assert EXIT_UNEXPECTED not in seen_codes


def test_custom_exit_code_override() -> None:
    err = DataPipelineError("custom failure", exit_code=42)
    assert err.exit_code == 42
    assert str(err) == "custom failure"


def test_exit_code_table_entries() -> None:
    codes = [row[0] for row in EXIT_CODE_TABLE]
    names = [row[1] for row in EXIT_CODE_TABLE]
    assert codes == [0, 2, 3, 4, 5, 6, 7, 8, 99]
    assert "EXIT_SUCCESS" in names
    assert "EXIT_UNEXPECTED" in names
    assert len(EXIT_CODE_TABLE) == 9


def test_format_cli_error() -> None:
    err = MissingColumnError("Column\t'target_bin'   missing")
    formatted = format_cli_error(err)
    assert formatted == "[MissingColumnError code=6] Column 'target_bin' missing"
