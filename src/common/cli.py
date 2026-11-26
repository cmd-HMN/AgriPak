import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.common.core.errors import (
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    InvalidArgumentError,
    PipelineError,
    format_cli_error,
)
from src.common.core.logging import configure_logging, get_console


class CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise InvalidArgumentError(message)


@dataclass(frozen=True)
class CLISpec:
    prog: str
    description: str
    default_input: str
    default_output: str | None = None
    default_log_dir: str = "log"


def add_shared_arguments(parser: argparse.ArgumentParser, spec: CLISpec) -> None:
    parser.add_argument("--input", type=Path, default=Path(spec.default_input))
    if spec.default_output is not None:
        parser.add_argument(
            "--output",
            "--output-dir",
            dest="output",
            type=Path,
            default=Path(spec.default_output),
        )
    parser.add_argument("--log-dir", type=Path, default=Path(spec.default_log_dir))
    parser.add_argument("--debug", action="store_true", default=False)
    if "-h" not in parser._option_string_actions:
        parser.add_argument("-h", "--help", action="store_true", dest="show_help")


def build_cli_parser(
    spec: CLISpec,
    configure_extra: Callable[[argparse.ArgumentParser], None] | None = None,
) -> CLIArgumentParser:
    parser = CLIArgumentParser(
        prog=spec.prog, description=spec.description, add_help=False
    )
    add_shared_arguments(parser, spec)
    if configure_extra is not None:
        configure_extra(parser)
    return parser


def print_summary_table(
    console: Console,
    title: str,
    rows: Sequence[tuple[str, str]],
) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)


def print_cli_help(parser: argparse.ArgumentParser, console: Console) -> None:
    console.print(Panel(parser.format_help(), title=parser.prog, border_style="cyan"))


def run_with_progress(
    console: Console,
    total: float,
    description: str,
    runner: Callable[[Any], None],
) -> None:
    _ = console
    total_f = max(1.0, float(total))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(stderr=True),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=total_f)

        def advance(stage: str, weight: float = 1.0) -> None:
            progress.update(
                task_id, advance=weight, description=f"{description}: {stage}"
            )

        runner(advance)
        progress.update(task_id, completed=total_f)


def handle_cli_exception(
    exc: Exception,
    err_console: Console,
    logger: logging.Logger | None,
    debug: bool,
) -> int:
    if isinstance(exc, PipelineError):
        err_console.print(f"[bold red]{format_cli_error(exc)}[/bold red]")
        if logger is not None:
            logger.error(format_cli_error(exc))
        return exc.exit_code
    err_console.print(
        f"[bold red][UnexpectedError code={EXIT_UNEXPECTED}] {exc}[/bold red]"
    )
    if logger is not None:
        logger.exception("Unhandled exception", exc_info=exc if debug else None)
    return EXIT_UNEXPECTED


def run_cli_app(
    spec: CLISpec,
    argv: Sequence[str] | None,
    execute: Callable[[argparse.Namespace, Console, logging.Logger], int],
    configure_extra: Callable[[argparse.ArgumentParser], None] | None = None,
    logger_name: str = "agripak",
) -> int:
    out_console = get_console(stderr=False)
    err_console = get_console(stderr=True)
    logger: logging.Logger | None = None
    debug = False
    try:
        parser = build_cli_parser(spec, configure_extra=configure_extra)
        args = parser.parse_args(list(argv) if argv is not None else None)
        debug = bool(getattr(args, "debug", False))
        if getattr(args, "show_help", False):
            print_cli_help(parser, out_console)
            return EXIT_SUCCESS
        logger = configure_logging(
            log_dir=getattr(args, "log_dir", Path(spec.default_log_dir)),
            logger_name=logger_name,
            debug=debug,
            console=err_console,
        )
        return execute(args, out_console, logger)
    except Exception as exc:
        return handle_cli_exception(exc, err_console, logger, debug)
