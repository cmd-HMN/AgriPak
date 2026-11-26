import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.common.core.io import ensure_directory


def get_console(stderr: bool = False) -> Console:
    return Console(stderr=stderr)


def configure_logging(
    log_dir: Path | None = None,
    logger_name: str = "agripak",
    filename: str = "agripak.log",
    debug: bool = False,
    console: Console | None = None,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    rich_handler = RichHandler(
        console=console if console is not None else Console(stderr=True),
        rich_tracebacks=debug,
        show_path=debug,
        markup=True,
        omit_repeated_times=False,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)
    if log_dir is not None:
        target_dir = ensure_directory(Path(log_dir))
        file_handler = logging.FileHandler(target_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    return logger
