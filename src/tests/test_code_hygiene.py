import ast
from pathlib import Path
import tokenize


def _data_package_py_files() -> list[Path]:
    src_dir = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for sub in ("data", "tests"):
        files.extend(
            p for p in (src_dir / sub).rglob("*.py") if "__pycache__" not in p.parts
        )
    return sorted(files)


def test_zero_comments_in_src_data() -> None:
    offenders: list[str] = []
    for py_file in _data_package_py_files():
        with tokenize.open(py_file) as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type == tokenize.COMMENT:
                    offenders.append(f"{py_file}:{token.start[0]}")
    assert offenders == []


def test_zero_docstrings_in_src_data() -> None:
    offenders: list[str] = []
    for py_file in _data_package_py_files():
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node, clean=False) is not None:
                    lineno = getattr(node, "lineno", 1)
                    offenders.append(f"{py_file}:{lineno}")
    assert offenders == []
