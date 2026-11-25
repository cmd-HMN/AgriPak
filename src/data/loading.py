from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.errors import (
    EmptyDataError,
    InputNotFoundError,
    InvalidArgumentError,
    MalformedDataError,
    ReportWriteError,
)
from src.data.utils import ensure_directory

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv", ".json", ".parquet"})


def _validate_non_empty_file(file_path: Path) -> None:
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        raise InputNotFoundError(
            f"Cannot access input file '{file_path}': {exc}"
        ) from exc
    if size == 0:
        raise EmptyDataError(f"Input file is empty (0 bytes): '{file_path}'")


def _read_single_file(file_path: Path, encoding: str, raw_text: bool) -> pd.DataFrame:
    _validate_non_empty_file(file_path)
    ext = file_path.suffix.lower()
    try:
        if ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            kwargs: dict[str, Any] = {"sep": sep, "encoding": encoding}
            if raw_text:
                kwargs.update(
                    {"dtype": str, "keep_default_na": False, "na_filter": False}
                )
            df = pd.read_csv(file_path, **kwargs)
        elif ext == ".json":
            if raw_text:
                df = pd.read_json(
                    file_path, encoding=encoding, dtype=cast(Any, False)
                ).astype(str)
            else:
                df = pd.read_json(file_path, encoding=encoding)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            if raw_text:
                df = df.astype(str)
        else:
            raise MalformedDataError(
                f"Unsupported file extension '{ext}' for '{file_path}'"
            )
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


def save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    target = Path(output_path)
    ext = target.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidArgumentError(
            f"Unsupported output extension '{ext}' for '{target}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    ensure_directory(target.parent)
    try:
        if ext == ".csv":
            df.to_csv(target, index=False)
        elif ext == ".tsv":
            df.to_csv(target, sep="\t", index=False)
        elif ext == ".json":
            df.to_json(target, orient="records", indent=2)
        elif ext == ".parquet":
            df.to_parquet(target, index=False)
    except OSError as exc:
        raise ReportWriteError(
            f"Failed to write DataFrame to '{target}': {exc}"
        ) from exc
    return target
