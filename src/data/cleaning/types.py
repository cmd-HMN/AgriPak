from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.core.validation import to_float, to_int
from src.common.dataset.columns import DatasetSchema
from src.common.report.markdown import dataframe_to_markdown


@dataclass
class CleaningEntry:
    step: str
    issue: str
    action: str
    rows_affected: int


@dataclass
class CleaningAudit:
    entries: list[CleaningEntry] = field(default_factory=list)

    def add(self, step: str, issue: str, action: str, rows_affected: int) -> None:
        self.entries.append(CleaningEntry(step, issue, action, int(rows_affected)))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["step", "issue", "action", "rows_affected"])
        return pd.DataFrame(
            [
                {
                    "step": e.step,
                    "issue": e.issue,
                    "action": e.action,
                    "rows_affected": e.rows_affected,
                }
                for e in self.entries
            ]
        )

    def to_markdown(self) -> str:
        return dataframe_to_markdown(self.to_dataframe())


@dataclass
class CleaningResult:
    data: pd.DataFrame
    audit: CleaningAudit
    schema: DatasetSchema
    rows_before: int
    rows_after: int
    memory_mb_before: float
    memory_mb_after: float


def _memory_usage_mb(df: pd.DataFrame) -> float:
    return round(to_float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0), 4)


def _compact_int_dtype(series: pd.Series) -> str:
    if bool(series.isna().any()):
        return "Int64"
    lo, hi = to_int(series.min()), to_int(series.max())
    for name, np_type in (("int8", np.int8), ("int16", np.int16), ("int32", np.int32)):
        info = np.iinfo(np_type)
        if lo >= info.min and hi <= info.max:
            return name
    return "int64"
