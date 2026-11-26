import pytest

from src.common.core.errors import (
    EXIT_CODE_TABLE,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    EmptyDataError,
    InputNotFoundError,
    InvalidArgumentError,
    MalformedDataError,
    MissingColumnError,
    PipelineError,
    ReportWriteError,
    format_cli_error,
)


@pytest.mark.parametrize(
    ("exc_cls", "expected_code"),
    [
        (InvalidArgumentError, 2),
        (InputNotFoundError, 3),
        (EmptyDataError, 4),
        (MalformedDataError, 5),
        (MissingColumnError, 6),
        (ReportWriteError, 8),
        (PipelineError, 99),
    ],
)
def test_pipeline_error_exit_codes(
    exc_cls: type[PipelineError], expected_code: int
) -> None:
    err = exc_cls("  something   went\nwrong  ")
    assert err.exit_code == expected_code
    assert err.message == "something went wrong"
    assert str(err) == "something went wrong"


def test_pipeline_error_custom_exit_code() -> None:
    err = PipelineError("custom", exit_code=42)
    assert err.exit_code == 42
    assert format_cli_error(err) == "[PipelineError code=42] custom"


def test_format_cli_error_normalizes_whitespace() -> None:
    err = InvalidArgumentError("bad\t\ninput")
    assert format_cli_error(err) == "[InvalidArgumentError code=2] bad input"


def test_exit_code_table_covers_all_codes() -> None:
    codes = [row[0] for row in EXIT_CODE_TABLE]
    assert codes == [0, 2, 3, 4, 5, 6, 7, 8, 9, 99]
    assert EXIT_SUCCESS == 0
    assert EXIT_UNEXPECTED == 99
