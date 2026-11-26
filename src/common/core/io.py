from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.common.core.errors import (
    EmptyDataError,
    InputNotFoundError,
    InvalidArgumentError,
    MalformedDataError,
    ReportWriteError,
)
from src.common.core.validation import ensure_not_overwriting_input

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv", ".json", ".parquet"})

_WRITERS: dict[str, Callable[[pd.DataFrame, Path], Any]] = {
    ".csv": lambda df, path: df.to_csv(path, index=False),
    ".tsv": lambda df, path: df.to_csv(path, sep="\t", index=False),
    ".json": lambda df, path: df.to_json(path, orient="records", indent=2),
    ".parquet": lambda df, path: df.to_parquet(path, index=False),
}


def ensure_directory(path: Path) -> Path:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReportWriteError(f"Unable to create directory '{target}': {exc}") from exc
    return target


def _validate_non_empty_file(file_path: Path) -> None:
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise InputNotFoundError(
            f"Cannot access input file '{file_path}': {exc}"
        ) from exc
    if size == 0:
        raise EmptyDataError(f"Input file is empty (0 bytes): '{file_path}'")


def _read_delimited(
    file_path: Path, ext: str, encoding: str, raw_text: bool
) -> pd.DataFrame:
    sep = "\t" if ext == ".tsv" else ","
    kwargs: dict[str, Any] = {"sep": sep, "encoding": encoding}
    if raw_text:
        kwargs.update({"dtype": str, "keep_default_na": False, "na_filter": False})
    return pd.read_csv(file_path, **kwargs)


def _read_structured(
    file_path: Path, ext: str, encoding: str, raw_text: bool
) -> pd.DataFrame:
    if ext == ".json":
        if raw_text:
            return pd.read_json(
                file_path, encoding=encoding, dtype=cast(Any, False)
            ).astype(str)
        return pd.read_json(file_path, encoding=encoding)
    if ext == ".parquet":
        df = pd.read_parquet(file_path)
        return df.astype(str) if raw_text else df
    raise MalformedDataError(f"Unsupported file extension '{ext}' for '{file_path}'")


def _read_single_file(file_path: Path, encoding: str, raw_text: bool) -> pd.DataFrame:
    _validate_non_empty_file(file_path)
    ext = file_path.suffix.lower()
    try:
        if ext in {".csv", ".tsv"}:
            df = _read_delimited(file_path, ext, encoding, raw_text)
        else:
            df = _read_structured(file_path, ext, encoding, raw_text)
    except (EmptyDataError, MalformedDataError):
        raise
    except pd.errors.EmptyDataError as exc:
        raise EmptyDataError(f"Input file contains no data: '{file_path}'") from exc
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError, OSError) as exc:
        raise MalformedDataError(f"Failed to parse '{file_path}': {exc}") from exc
    if df.empty or len(df.columns) == 0:
        raise EmptyDataError(f"Loaded dataset has 0 rows or 0 columns: '{file_path}'")
    return df


def _load_frames(input_path: Path, encoding: str, raw_text: bool) -> pd.DataFrame:
    files = resolve_input_files(Path(input_path))
    frames = [
        _read_single_file(fp, encoding=encoding, raw_text=raw_text) for fp in files
    ]
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if combined.empty or len(combined.columns) == 0:
        raise EmptyDataError(f"Combined dataset from '{input_path}' is empty")
    return combined


def resolve_input_files(input_path: Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise InputNotFoundError(f"Input path does not exist: '{path}'")
    if path.is_file():
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise MalformedDataError(
                f"Unsupported file extension '{ext}' for '{path}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [path]
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise InputNotFoundError(
                f"No supported data files found in directory: '{path}'"
            )
        return files
    raise InputNotFoundError(f"Input path is neither a file nor a directory: '{path}'")


def load_dataset(input_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    return _load_frames(input_path, encoding=encoding, raw_text=False)


def load_raw_text(input_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    return _load_frames(input_path, encoding=encoding, raw_text=True)


def save_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    source_input: Path | None = None,
) -> Path:
    target = Path(output_path)
    ensure_not_overwriting_input(target, source_input)
    ext = target.suffix.lower()
    writer = _WRITERS.get(ext)
    if writer is None:
        raise InvalidArgumentError(
            f"Unsupported output extension '{ext}' for '{target}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    ensure_directory(target.parent)
    try:
        writer(df, target)
    except OSError as exc:
        raise ReportWriteError(
            f"Failed to write DataFrame to '{target}': {exc}"
        ) from exc
    return target


def write_text_file(content: str, output_path: Path) -> Path:
    target = Path(output_path)
    ensure_directory(target.parent)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(f"Failed to write file '{target}': {exc}") from exc
    return target
