<div align="center">

# AgriPak

**Modular, schema-agnostic agricultural yield-risk data audit, cleaning, statistical analysis, and automated reporting.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 66 passing](https://img.shields.io/badge/tests-66%20passing-brightgreen.svg)](#testing)
[![Pyright: 0 errors](https://img.shields.io/badge/pyright-0%20errors-brightgreen.svg)](#testing)
[![Report: Dynamic](https://img.shields.io/badge/report-dynamic-blue.svg)](src/data/README.md)

**[Installation](#installation)** |
**[Getting Started](#getting-started)** |
**[Notebooks & Features](#notebooks--feature-selection)** |
**[Full Report](src/data/README.md)** |
**[Preprocessing Guide](src/preprocess/README.md)** |
**[CLI Reference](#cli-reference)** |
**[Testing](#testing)** |
**[Exit Codes](#exit-codes)** |
**[License](#license)**

</div>

---

AgriPak is a Python workspace organized into three focused packages (`src/common/` for shared schema/IO/errors/reporting, `src/data/` for data audit/cleaning/EDA, and `src/preprocess/` for feature engineering/transforms/splits):

- Schema-agnostic data auditing, cleaning, statistical exploration, leakage detection, and plot-driven feature engineering with dynamic column inference and `Unidentified` fallbacks.
- Automated generation of reproducible Markdown reports ([src/data/README.md](src/data/README.md) and [src/preprocess/README.md](src/preprocess/README.md)) and visual assets (`src/data/assets/`, `src/preprocess/assets/`) backed by typed CLIs and GitHub Actions CI.

> **Reports:** See [**src/data/README.md**](src/data/README.md) (Full Combined Report) and [**src/preprocess/README.md**](src/preprocess/README.md) (Preprocessing Guide) for the complete statistical audit, visualizations, leakage checks, and cross-validated model results.

## Installation

### Prerequisites

- Python `3.13` or higher
- `pip` or `uv`

### Using `venv` and `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn rich pytest
```

### Using `uv`

```bash
uv sync
```

---

## Getting Started

Run the full end-to-end data pipeline, preprocessing pipeline, and combined report assembler in order:

```bash
python -m src.data run --input data/raw --output src/data --assets src/data/assets
python -m src.preprocess run --input data/raw --output src/preprocess --assets src/preprocess/assets
python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
```

Regenerate only the reports and plots without writing intermediate CSVs:

```bash
python -m src.data report --input data/raw --output src/data --assets src/data/assets
python -m src.preprocess report --input data/raw --output src/preprocess --assets src/preprocess/assets
python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
```

---

## Notebooks & Feature Selection

Interactive notebooks split exploratory analysis and feature engineering into two focused stages:

| Notebook | Purpose | Output |
| --- | --- | --- |
| [`notebooks/data/d.ipynb`](notebooks/data/d.ipynb) | Data quality audit, EDA plots, non-parametric tests, 5-step cleaning, and leakage checks | `data/clean/crop_data_clean.csv` |
| [`notebooks/preprocess/p.ipynb`](notebooks/preprocess/p.ipynb) | Plot-driven feature engineering, 5-fold CV ROC-AUC validation, and final feature selection | `data/clean/crop_data_features.csv` |

### Selected Modeling Features (`8` kept, `5` dropped)

| Decision | Features | Reason |
| --- | --- | --- |
| **Keep — Raw signal (4)** | `rainfall_mm`, `fertilizer_used`, `irrigation_used`, `temperature_c` | Core pre-season drivers (`rainfall_mm` ablation drops 5-fold CV AUC from `0.978` to `0.756`); `temperature_c` carries a small but significant monotone effect (`p ≈ 0`). |
| **Keep — Engineered (4)** | `rain_deficit`, `rain_x_inputs`, `deficit_x_noinputs`, `inputs_used` | Captures the `800 mm` zero-risk floor (`rain_deficit` `ρ = +0.639`), input-driven curve shifts (`rain_x_inputs` `ρ = -0.630`), unmitigated drought risk (`deficit_x_noinputs` `ρ = +0.509`), and monotonic input stepping (`inputs_used` `64.3% → 7.3%`). Reaches `0.978` (`LogisticRegression`) and `0.977` (`HistGradientBoosting`) 5-fold CV ROC-AUC. |
| **Drop — Noise / Post-harvest (5)** | `days_to_harvest`, `crop`, `region`, `soil_type`, `weather_condition` | `days_to_harvest` has zero signal (`ρ = 0.003`, `p = 0.554`) and is post-harvest; the 4 categoricals sit flat at `35.0%` baseline (`Cramér's V ≤ 0.012`) and add `+0.000` CV AUC lift (`0.977` vs `0.977`). |

---

## CLI Reference

Run `python -m src.data --help` or `python -m src.preprocess --help` for full option details.

### `src.data` (7 commands, `--assets src/data/assets`)

| Command   | Default `--output` | Description                                                                          |
| --------- | ------------------ | ------------------------------------------------------------------------------------ |
| `run`     | `src/data`         | Execute the data pipeline and write cleaned CSV, plots, fragment, and report         |
| `audit`   | `src/data/assets`  | Infer dynamic schema and run data quality and text-hygiene audit (`quality_audit.md`) |
| `clean`   | `data/clean`       | Run the 5-step cleaning pipeline and export `cleaned_data.csv` + `cleaning_audit.md` |
| `analyze` | `src/data/assets`  | Run full statistical EDA and leakage check (`analysis_summary.md`)                   |
| `plot`    | `src/data/assets`  | Generate data EDA PNG charts into `src/data/assets/`                                 |
| `report`  | `src/data`         | Run analysis, generate data charts, and write `src/data/report_fragment.md`          |
| `help`    | —                  | Print Rich command table or help for a specific subcommand                           |

### `src.preprocess` (9 commands, `--assets src/preprocess/assets`)

| Command    | Default `--output`     | Description                                                                          |
| ---------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `run`      | `src/preprocess`       | Execute preprocessing pipeline, splits, CV benchmarks, plots, and guide              |
| `features` | `data/clean`           | Infer risk threshold and export `engineered_features.csv`                            |
| `encode`   | `data/clean`           | Fit/apply categorical encoding and export encoded features                           |
| `scale`    | `data/clean`           | Fit/apply numeric scaling and export scaled features                                 |
| `split`    | `data/clean`           | Generate stratified train/val/test splits                                            |
| `evaluate` | `src/preprocess/assets`| Run 5-fold CV ROC-AUC benchmarks and ablation checks                                 |
| `plot`     | `src/preprocess/assets`| Generate feature-engineering PNG charts into `src/preprocess/assets/`                |
| `report`   | `src/preprocess`       | Run benchmarks, generate charts, and write `src/preprocess/README.md` + fragment     |
| `help`     | —                      | Print Rich command table or help for a specific subcommand                           |

### `src.common.report` (Combined Report Assembler)

```bash
python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
```

### Common Flags

| Flag               | Default                   | Description                                                           |
| ------------------ | ------------------------- | --------------------------------------------------------------------- |
| `--input`          | `data/raw`                | Input dataset file or directory (`.csv`, `.tsv`, `.json`, `.parquet`) |
| `--output`         | Per command (table above) | Output directory for reports, audits, and exported CSVs               |
| `--assets`         | `src/assets`              | Output directory for generated PNG plots                              |
| `--log-dir`        | `log`                     | Directory for timestamped file logs (`log/agripak.log`, gitignored)   |
| `--target`         | `None`                    | Optional explicit target column (auto-inferred when omitted)          |
| `--min-n`          | `None`                    | Minimum subgroup size (defaults to `max(20, int(0.01 * n))`)          |
| `--n-splits`       | `5`                       | Number of Stratified K-Fold cross-validation splits                   |
| `--leak-threshold` | `0.90`                    | Single-feature association leakage threshold                          |
| `--seed`           | `42`                      | Random seed for split replication and cross-validation                |
| `--debug`          | `False`                   | Print full tracebacks to `stderr` on failure                          |

---

## Exit Codes

| Code | Error Type | Meaning |
| --- | --- | --- |
| `0` | — | Success |
| `2` | `CLIUsageError` | Invalid CLI arguments or flags |
| `3` | `DataLoadError` | Missing, unreadable, or empty input dataset |
| `4` | `SchemaError` | Required columns missing or invalid |
| `5` | `ValidationError` | Data or parameter validation failure |
| `6` | `LeakageError` | Feature-target leakage threshold exceeded |
| `7` | `PlotError` | Chart rendering or export failure |
| `8` | `ReportError` | Markdown report generation or assembly failure |
| `9` | `TransformError` | Feature engineering, encoding, scaling, or split failure |
| `99` | `Exception` | Unexpected unhandled error |

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

Static type check:

```bash
uvx pyright --pythonpath .venv/bin/python src/
```

## License

AgriPak is MIT-licensed, as found in the [LICENSE](LICENSE) file.
