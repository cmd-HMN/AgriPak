<div align="center">

# AgriPak

**Low-yield risk prediction for agricultural fields — schema-agnostic data cleaning, leakage auditing, feature engineering, and baseline modeling.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 72 passing](https://img.shields.io/badge/tests-72%20passing-brightgreen.svg)](#testing)
[![Pyright: 0 errors](https://img.shields.io/badge/pyright-0%20errors-brightgreen.svg)](#testing)

**[More About AgriPak](#more-about-agripak)** |
**[Installation](#installation)** |
**[Getting Started](#getting-started)** |
**[Results](#results)** |
**[CLI Reference](#cli-reference)** |
**[Exit Codes](#exit-codes)** |
**[Testing](#testing)** |
**[License](#license)**

</div>

---

AgriPak is a Python project that provides two high-level capabilities:

- Schema-agnostic data auditing, 5-step cleaning, leakage detection, plot-driven feature engineering, and reproducible stratified 80/20 splitting with `Unidentified` fallbacks.
- Holdout Logistic Regression baseline modeling, cross-validated benchmarks, and automated Markdown/PNG report generation backed by typed Rich CLIs and GitHub Actions CI.

## More About AgriPak

| Resource | Description |
| --- | --- |
| [Full Data Analysis Report](src/data/README.md) | Dynamically generated statistical audit, cleaning log, leakage check, and visualizations |
| [Preprocessing Guide](src/preprocess/README.md) | Feature engineering, encoding, scaling, signal selection, and CV CLI guide |
| [Common Utilities](src/common/README.md) | Shared schema inference, IO, centralized error codes, and report assembler |
| [EDA Notebook](notebooks/data/d.ipynb) | Interactive data audit, cleaning, and leakage analysis (`data/clean/crop_data_clean.csv`) |
| [Feature Engineering Notebook](notebooks/preprocess/p.ipynb) | Plot-driven feature engineering, CV validation, and feature selection (`data/clean/crop_data_features.csv`) |
| [Baseline Model Notebook](notebooks/models/m.ipynb) | Reproducible 80/20 stratified Logistic Regression baseline and confusion matrix |

---

## Installation

### Prerequisites

- Python `3.13` or higher
- `uv` (recommended) or `pip`

### Using `uv`

```bash
uv sync
```

### Using `venv` and `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn rich pytest
```

---

## Getting Started

Run the full pipeline in order — clean and audit data, generate reproducible 80/20 stratified splits, evaluate the Logistic Regression baseline, and assemble the combined report:

```bash
uv run python -m src.data run --input data/raw --output src/data --assets src/data/assets
uv run python -m src.preprocess split --input data/clean/crop_data_features.csv --output data/splits --target low_yield_risk --seed 42 --test-size 0.2
uv run python -m src.model baseline --input data/splits --target low_yield_risk
uv run python -m src.preprocess report --input data/raw --output src/preprocess --assets src/preprocess/assets
uv run python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
```

---

## Results

### Cleaning Decisions (`data/raw/crop_data.csv`, `50,000` rows × `10` cols)

1. **Text normalization:** Stripped leading/trailing whitespace and scanned missing placeholder tokens (`0` issues found).
2. **Type coercion:** Verified numeric columns (`rainfall_mm`, `temperature_c`, `days_to_harvest`) and standardized binary flags (`fertilizer_used`, `irrigation_used`, `low_yield_risk`) to `0/1`.
3. **Range validation:** Verified `0` `±inf` values across numeric features.
4. **Missing values & duplicates:** Verified `0` missing values and `0` exact duplicate rows (`50,000` rows preserved; fallback policy drops missing targets and imputes numeric median, binary mode, and categorical `Unidentified`).
5. **Dtype compaction:** Cast categoricals to `category`, continuous numerics to `float64`, discrete integers to `int16`, and binary flags to `int8`, reducing memory footprint from `12.59 MB` to `1.19 MB`.

### Baseline Model Scores (80/20 Stratified Split, `seed=42`)

Reproducible split (`40,000` train / `10,000` test, both `35.0%` positive `low_yield_risk`) saved to `data/splits/train.csv` and `data/splits/test.csv`. `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)` fit strictly on `train.csv`:

| Evaluation Run | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src.model baseline` CLI (`data/splits`) | `13` | `0.9192` | `0.8895` | `0.8783` | `0.8838` | `0.9774` |
| `notebooks/models/m.ipynb` (4 raw signals) | `4` | `0.9194` | `0.8900` | `0.8783` | `0.8841` | `0.9774` |
| `notebooks/models/m.ipynb` (8 selected) | `8` | `0.9196` | `0.8896` | `0.8794` | `0.8845` | `0.9774` |

### Leakage Sanity Check (`|correlation| > 0.90` Threshold)

- **Columns flagged above `0.90`:** **None (`0` columns)**.
- **Strongest raw predictor:** `rainfall_mm` (`|ρ| = 0.636`, single-feature AUC `0.885`) — genuine pre-season physical driver, not leakage.
- **Temporal / post-harvest check:** `days_to_harvest` (`|ρ| = 0.003`, `p = 0.554`) is only known at or after harvest and carries zero signal, so it is excluded from pre-season modeling features.
- **Feature selection (`8` kept / `5` dropped):** Keeps `rainfall_mm`, `temperature_c`, `fertilizer_used`, `irrigation_used` + engineered `rain_deficit`, `rain_x_inputs`, `deficit_x_noinputs`, `inputs_used`; drops `days_to_harvest`, `crop`, `region`, `soil_type`, `weather_condition` (`+0.000` CV AUC lift).

---

## CLI Reference

Run `python -m <package> --help` for full option details.

### `src.data`

| Command | Description |
| --- | --- |
| `run` | Execute full data pipeline (clean, analyze, 14 plots, fragment, combined report) |
| `audit` | Infer dynamic schema and run data quality and text-hygiene audit |
| `clean` | Run 5-step cleaning pipeline and export `cleaned_data.csv` |
| `analyze` | Run statistical EDA and leakage check |
| `plot` | Generate 14 EDA PNG charts into `--assets` (`src/data/assets`) |
| `report` | Generate charts, `src/data/report_fragment.md`, and assemble `src/data/README.md` |
| `help` | Print Rich CLI help panel |

### `src.preprocess`

| Command | Description |
| --- | --- |
| `run` | Execute full preprocessing pipeline, CV evaluation, plots, and reports |
| `transform` | Engineer domain features and export `engineered.csv` |
| `encode` | Fit/apply categorical encoding (`onehot` or `ordinal`) |
| `scale` | Fit/apply numeric scaling (`standard`, `robust`, or `minmax`) |
| `select` | Filter signal vs noise features via Spearman and Cramér's V |
| `split` | Create reproducible stratified 80/20 `train.csv` and `test.csv` in `data/splits` |
| `evaluate` | Run 5-fold Stratified K-Fold CV ROC-AUC benchmarks and ablation |
| `plot` | Generate 4 preprocessing PNG charts into `--assets` (`src/preprocess/assets`) |
| `report` | Generate charts, `src/preprocess/README.md`, fragment, and assemble combined report |
| `help` | Print Rich CLI help panel |

### `src.model`

| Command | Description |
| --- | --- |
| `baseline` | Train Logistic Regression on `train.csv` and report Accuracy, Precision, Recall, F1, ROC-AUC |
| `help` | Print Rich CLI help panel |

### `src.common.report`

```bash
uv run python -m src.common.report --data-fragment src/data/report_fragment.md --preprocess-fragment src/preprocess/report_fragment.md --output src/data/README.md
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

## Testing

All tests use deterministic synthetic fixtures and run completely offline:

```bash
uv run python -m pytest src/tests -v
```

Static type check and lint:

```bash
uvx pyright --pythonpath .venv/bin/python src/common src/data src/preprocess src/model src/tests
uvx ruff check src/common src/data src/preprocess src/model src/tests
```

---

## License

AgriPak is MIT-licensed, as found in the [LICENSE](LICENSE) file.
