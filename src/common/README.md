<div align="center">

# AgriPak Common

**Shared schema inference, dataset IO, validation, error handling, plotting, and report assembly.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

**[Getting Started](#getting-started)** |
**[Report Assembler](#report-assembler)** |
**[Exit Codes](#exit-codes)** |
**[Project README](../../README.md)** |
**[License](#license)**

</div>

---

AgriPak Common provides the shared foundation used across `src.data` and `src.preprocess`:

- Schema-agnostic column inference with `Unidentified` fallbacks, multi-format dataset IO (`.csv`, `.tsv`, `.json`, `.parquet`), boundary validation, and centralized typed errors.
- Shared Rich/argparse CLI harness, headless matplotlib plotting helpers, Markdown table renderers, and combined report assembly.

---

## Getting Started

```python
from pathlib import Path
from src.common import infer_schema, load_dataset

df, source_path = load_dataset(Path("data/raw"))
schema = infer_schema(df)
```

---

## Report Assembler

Merge the `src.data` and `src.preprocess` report fragments into the combined report:

```bash
python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
```

---

## Exit Codes

| Code | Error Class | Meaning |
| --- | --- | --- |
| `0` | `EXIT_SUCCESS` | Pipeline completed successfully |
| `2` | `InvalidArgumentError` | Invalid CLI or function argument |
| `3` | `InputNotFoundError` | Input file or directory not found |
| `4` | `EmptyDataError` | Input dataset is empty or has no remaining rows |
| `5` | `MalformedDataError` | Input file is malformed or unsupported |
| `6` | `MissingColumnError` | Required column is missing from dataset |
| `7` | `AnalysisError` | Statistical or modeling analysis failed |
| `8` | `ReportWriteError` | Failed to write output or create directory |
| `9` | `TransformError` | Preprocessing or feature transformation failed |
| `99` | `EXIT_UNEXPECTED` | Unexpected unhandled error |

---

## License

AgriPak is MIT-licensed, as found in the [LICENSE](../../LICENSE) file.
