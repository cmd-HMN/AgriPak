EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 99


def _clean_message(message: str) -> str:
    return " ".join(str(message).split())


class PipelineError(Exception):
    exit_code: int = EXIT_UNEXPECTED

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        cleaned = _clean_message(message)
        self.message: str = cleaned
        self.exit_code: int = (
            type(self).exit_code if exit_code is None else int(exit_code)
        )
        super().__init__(cleaned)


class InvalidArgumentError(PipelineError):
    exit_code: int = 2


class InputNotFoundError(PipelineError):
    exit_code: int = 3


class EmptyDataError(PipelineError):
    exit_code: int = 4


class MalformedDataError(PipelineError):
    exit_code: int = 5


class MissingColumnError(PipelineError):
    exit_code: int = 6


class AnalysisError(PipelineError):
    exit_code: int = 7


class ReportWriteError(PipelineError):
    exit_code: int = 8


class TransformError(PipelineError):
    exit_code: int = 9


DataPipelineError = PipelineError


EXIT_CODE_TABLE: tuple[tuple[int, str, str], ...] = (
    (EXIT_SUCCESS, "EXIT_SUCCESS", "Pipeline completed successfully"),
    (
        InvalidArgumentError.exit_code,
        "InvalidArgumentError",
        "Invalid CLI or function argument",
    ),
    (
        InputNotFoundError.exit_code,
        "InputNotFoundError",
        "Input file or directory not found",
    ),
    (
        EmptyDataError.exit_code,
        "EmptyDataError",
        "Input dataset is empty or has no remaining rows",
    ),
    (
        MalformedDataError.exit_code,
        "MalformedDataError",
        "Input file is malformed or unsupported",
    ),
    (
        MissingColumnError.exit_code,
        "MissingColumnError",
        "Required column is missing from dataset",
    ),
    (
        AnalysisError.exit_code,
        "AnalysisError",
        "Statistical or modeling analysis failed",
    ),
    (
        ReportWriteError.exit_code,
        "ReportWriteError",
        "Failed to write output or create directory",
    ),
    (
        TransformError.exit_code,
        "TransformError",
        "Preprocessing or feature transformation failed",
    ),
    (EXIT_UNEXPECTED, "EXIT_UNEXPECTED", "Unexpected unhandled error"),
)


def format_cli_error(error: PipelineError) -> str:
    return f"[{type(error).__name__} code={error.exit_code}] {_clean_message(error.message)}"
