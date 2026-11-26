<div align="center">

# AgriPak Preprocess

**Dynamic feature engineering, encoding, scaling, signal selection, and cross-validated model evaluation.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

**[Getting Started](#getting-started)** |
**[Commands](#commands)** |
**[Features](#features)** |
**[Exit Codes](#exit-codes)** |
**[Full Report](../data/README.md)** |
**[Project README](../../README.md)** |
**[License](#license)**

</div>

---

AgriPak Preprocess provides two high-level capabilities:

- Plot-driven dynamic feature engineering, categorical encoding, numeric scaling, and Spearman / Cramér's V signal-vs-noise selection with `Unidentified` fallbacks.
- Stratified K-Fold cross-validation benchmarks (`LogisticRegression` and `HistGradientBoostingClassifier`), ablation checks, PNG chart generation, and automated Markdown reporting.

---

## Getting Started

Run the full preprocessing pipeline, generate charts, and write the report:

```bash
python -m src.preprocess run --input data/raw --output src/preprocess --assets src/preprocess/assets
```

Regenerate only the preprocess report, fragment, and charts:

```bash
python -m src.preprocess report --input data/raw --output src/preprocess --assets src/preprocess/assets
```

---

## Commands

| Command | Description |
| --- | --- |
| `run` | Execute full preprocessing pipeline, CV evaluation, plots, and reports |
| `transform` | Engineer domain features and export `engineered.csv` |
| `encode` | Fit and apply categorical encoding (`onehot` or `ordinal`) |
| `scale` | Fit and apply numeric scaling (`standard`, `robust`, or `minmax`) |
| `select` | Filter signal vs noise features and export `selected_features.csv` + `feature_manifest.csv` |
| `evaluate` | Run 5-fold Stratified K-Fold CV ROC-AUC benchmarks and ablation |
| `plot` | Generate preprocessing PNG charts into `--assets` |
| `report` | Generate `README.md`, `report_fragment.md`, and assemble combined report |
| `help` | Print CLI command table and options |

Key flags: `--input` (`data/raw`), `--output` (`src/preprocess`), `--assets` (`src/preprocess/assets`), `--scale-method` (`standard`), `--encode-method` (`onehot`), `--combined-report`, `--log-dir` (`log`), `--debug`.

---

## Features

| Feature | Definition | Purpose |
| --- | --- | --- |
| `binary_signal_sum` | Sum of active binary management signals (`0 / 1 / 2`) | Captures monotonic risk reduction across combined inputs |
| `primary_driver_deficit` | `max(0, threshold - primary_driver)` | Linearizes the inferred risk-floor boundary |
| `deficit_x_zero_signal` | `primary_driver_deficit * (binary_signal_sum == 0)` | Isolates unmitigated drought-risk regime |
| `primary_x_binary_sum` | `primary_driver * binary_signal_sum` | Encodes curve shift across input levels |

Unidentifiable columns fall back to `Unidentified` without crashing.

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
| `8` | `ReportWriteError` | Failed to write output or create directory |
| `9` | `TransformError` | Preprocessing or feature transformation failed |
| `99` | `EXIT_UNEXPECTED` | Unexpected unhandled error |

---

## License

AgriPak is MIT-licensed, as found in the [LICENSE](../../LICENSE) file.
