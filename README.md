<div align="center">

# AgriPak

**Modular, schema-agnostic agricultural yield-risk data audit, cleaning, statistical analysis, and automated reporting.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 63 passing](https://img.shields.io/badge/tests-63%20passing-brightgreen.svg)](#testing)
[![Report: Dynamic](https://img.shields.io/badge/report-dynamic-blue.svg)](src/data/README.md)

**[Installation](#installation)** |
**[Getting Started](#getting-started)** |
**[Full Report](src/data/README.md)** |
**[CLI Reference](#cli-reference)** |
**[Testing](#testing)** |
**[Exit Codes](#exit-codes)** |
**[License](#license)**

</div>

---

AgriPak is a Python package that provides two high-level features:

- Schema-agnostic data auditing, cleaning, statistical exploration, leakage detection, and plot-driven feature engineering with dynamic column inference and `Unidentified` fallbacks.
- Automated generation of reproducible Markdown reports ([src/data/README.md](src/data/README.md)) and visual assets ([src/assets/](src/assets/)) backed by a typed CLI and GitHub Actions CI.

> **Full Data Analysis Report:** See [**src/data/README.md**](src/data/README.md) for the complete generated statistical audit, 15 visualizations, leakage checks, and cross-validated model results.

---

## More About AgriPak

| Component | Path | Purpose |
| --- | --- | --- |
| Core package | [src/data/](src/data/) | Loading, schema inference, cleaning, analysis, plotting, reporting, CLI, errors |
| Generated report | [src/data/README.md](src/data/README.md) | Dynamically generated full analysis report |
| Visual assets | [src/assets/](src/assets/) | 15 PNG charts embedded in the report |
| Test suite | [src/tests/](src/tests/) | Offline `pytest` suite with synthetic fixtures and code-hygiene checks |
| CI workflow | [data-report.yml](.github/workflows/data-report.yml) | Runs tests and regenerates report on `data/raw/**` changes |

---

## Installation

### Prerequisites

- Python `3.13` or higher
- `pip` or `uv`

### Using `venv` and `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn pytest
```

### Using `uv`

```bash
uv sync
```

---

## Getting Started

Run the full end-to-end pipeline (load, clean, analyze, engineer features, generate plots, and write the report):

```bash
python -m src.data run --input data/raw --output src/data --assets src/assets
```

Regenerate only the report and plots in `src/data/README.md` and `src/assets/` without writing intermediate CSVs:

```bash
python -m src.data report --input data/raw --output src/data --assets src/assets
```

---

## CLI Reference

Run `python -m src.data --help` or `python -m src.data <command> --help` for full option details.

| Command | Description |
| --- | --- |
| `run` | Execute the full pipeline and write cleaned CSV, engineered CSV, plots, and report |
| `audit` | Infer dynamic schema and run data quality and text-hygiene audit |
| `clean` | Run the 5-step cleaning pipeline and export `cleaned_data.csv` + `cleaning_audit.md` |
| `analyze` | Run full statistical EDA, leakage check, and cross-validated model evaluation |
| `features` | Infer risk threshold and export `engineered_features.csv` |
| `plot` | Generate all 15 exploratory and feature-engineering PNG charts |
| `report` | Run analysis, generate all charts, and write `src/data/README.md` |

### Individual Step Examples

```bash
python -m src.data audit --input data/raw --output src/data
python -m src.data clean --input data/raw --output src/data
python -m src.data analyze --input data/raw --output src/data
python -m src.data features --input data/raw --output src/data
python -m src.data plot --input data/raw --assets src/assets
```

### Common Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | `data/raw` | Input dataset file or directory (`.csv`, `.tsv`, `.json`, `.parquet`) |
| `--output` | `src/data` | Output directory for reports, audits, and exported CSVs |
| `--assets` | `src/assets` | Output directory for generated PNG plots |
| `--target` | `None` | Optional explicit target column (auto-inferred when omitted) |
| `--min-n` | `None` | Minimum subgroup size (defaults to `max(20, int(0.01 * n))`) |
| `--n-splits` | `5` | Number of Stratified K-Fold cross-validation splits |
| `--leak-threshold` | `0.90` | Single-feature association leakage threshold |
| `--seed` | `42` | Random seed for split replication and cross-validation |
| `--debug` | `False` | Print full tracebacks to `stderr` on failure |

---

## Testing

All tests use deterministic synthetic fixtures and run completely offline:

```bash
python -m pytest src/tests -v
```

With `uv`:

```bash
uv run --no-sync --with pytest python -m pytest src/tests -v
```

---

## Exit Codes

| Code | Error Class | Description |
| --- | --- | --- |
| `0` | `EXIT_SUCCESS` | Pipeline completed successfully |
| `2` | `InvalidArgumentError` | Invalid CLI or function argument |
| `3` | `InputNotFoundError` | Input file or directory not found |
| `4` | `EmptyDataError` | Input dataset is empty or has no remaining rows |
| `5` | `MalformedDataError` | Input file is malformed or unsupported |
| `6` | `MissingColumnError` | Required column is missing from dataset |
| `7` | `AnalysisError` | Statistical or modeling analysis failed |
| `8` | `ReportWriteError` | Failed to write output or create directory |
| `99` | `EXIT_UNEXPECTED` | Unexpected unhandled error |

---

## License

AgriPak is MIT-licensed, as found in the [LICENSE](LICENSE) file.
