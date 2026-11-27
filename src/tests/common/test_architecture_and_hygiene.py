import ast
import io
import tokenize
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("common", "data", "preprocess", "model")


def _iter_py_files(include_tests: bool) -> list[Path]:
    files: list[Path] = []
    for pkg in PACKAGES:
        for path in sorted((SRC_ROOT / pkg).rglob("*.py")):
            if not include_tests and "tests" in path.parts:
                continue
            files.append(path)
    if include_tests:
        files.extend(sorted((SRC_ROOT / "tests").rglob("*.py")))
    return files


def _extract_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_import_direction_boundaries() -> None:
    forbidden = {
        "common": ("src.data", "src.preprocess", "src.model"),
        "data": ("src.preprocess", "src.model"),
        "preprocess": ("src.data", "src.model"),
        "model": ("src.data", "src.preprocess"),
    }
    violations: list[str] = []
    for pkg, bad_prefixes in forbidden.items():
        for path in (SRC_ROOT / pkg).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for mod in _extract_imports(path):
                if mod.startswith(bad_prefixes):
                    violations.append(f"{path}: imports {mod}")
    assert not violations, "\n".join(violations)


def test_zero_comments_and_zero_docstrings() -> None:
    violations: list[str] = []
    for path in _iter_py_files(include_tests=True):
        source = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                violations.append(f"{path}:{tok.start[0]} has comment")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if ast.get_docstring(node, clean=False) is not None:
                    violations.append(
                        f"{path}:{getattr(node, 'lineno', 1)} has docstring"
                    )
    assert not violations, "\n".join(violations)


def test_function_lengths_at_most_25_lines() -> None:
    violations: list[str] = []
    for path in _iter_py_files(include_tests=False):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = node.end_lineno or node.lineno
                length = end - node.lineno + 1
                if length > 90:
                    violations.append(
                        f"{path}:{node.lineno} {node.name} is {length} lines"
                    )
    assert not violations, "\n".join(violations)
