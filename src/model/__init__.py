from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.model.baseline.logistic import (
        BaselineMetrics,
        load_split_folder,
        run_baseline,
        train_evaluate_logistic,
    )
    from src.model.cli import main

_EXPORTS: dict[str, tuple[str, str]] = {
    "BaselineMetrics": ("src.model.baseline.logistic", "BaselineMetrics"),
    "load_split_folder": ("src.model.baseline.logistic", "load_split_folder"),
    "main": ("src.model.cli", "main"),
    "run_baseline": ("src.model.baseline.logistic", "run_baseline"),
    "train_evaluate_logistic": (
        "src.model.baseline.logistic",
        "train_evaluate_logistic",
    ),
}

__all__ = [
    "BaselineMetrics",
    "load_split_folder",
    "main",
    "run_baseline",
    "train_evaluate_logistic",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.model' has no attribute {name!r}")
    mod_path, attr = _EXPORTS[name]
    return getattr(import_module(mod_path), attr)
