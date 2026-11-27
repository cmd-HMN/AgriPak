<div align="center">

# AgriPak Model

**Baseline Logistic Regression training and holdout test evaluation on reproducible stratified splits.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)

**[Getting Started](#getting-started)** |
**[Commands](#commands)** |
**[Baseline Results](#baseline-results)** |
**[Exit Codes](#exit-codes)** |
**[Full Report](../data/README.md)** |
**[Project README](../../README.md)** |
**[License](#license)**

</div>

---

AgriPak Model provides two high-level capabilities:

- Leakage-free `StandardScaler` + `OneHotEncoder` + `LogisticRegression` training fit strictly on `train.csv` and evaluated on holdout `test.csv`.
- Rich CLI reporting of Accuracy, Precision, Recall, F1, ROC-AUC, and optional JSON metrics export.

---

## Getting Started

Run the Logistic Regression baseline on the reproducible 80/20 splits:

```bash
uv run python -m src.model baseline --input data/splits --target low_yield_risk
```

Export test metrics to JSON:

```bash
uv run python -m src.model baseline --input data/splits --target low_yield_risk --output data/splits/baseline_metrics.json
```

---

## Commands

| Command | Description |
| --- | --- |
| `baseline` | Load `train.csv` and `test.csv`, fit Logistic Regression on train, and report test metrics |
| `help` | Print Rich CLI help panel and options |

Key flags: `--input` (`data/splits`), `--target` (auto-inferred if omitted), `--seed` (`42`), `--output` / `--output-dir` (optional JSON export), `--log-dir` (`log`), `--debug`.

---

## Baseline Results

Evaluated on the reproducible 80/20 stratified split (`40,000` train / `10,000` test, `35.0%` positive `low_yield_risk`, `seed=42`):

| Run | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src.model baseline` CLI | `13` | `0.9192` | `0.8895` | `0.8783` | `0.8838` | `0.9774` |
| `notebooks/models/m.ipynb` (4 raw signals) | `4` | `0.9194` | `0.8900` | `0.8783` | `0.8841` | `0.9774` |
| `notebooks/models/m.ipynb` (8 selected) | `8` | `0.9196` | `0.8896` | `0.8794` | `0.8845` | `0.9774` |

---

## Exit Codes

| Code | Error Class | Meaning |
| --- | --- | --- |
| `0` | `EXIT_SUCCESS` | Pipeline completed successfully |
| `2` | `InvalidArgumentError` | Invalid CLI or function argument |
| `3` | `InputNotFoundError` | Split directory or `train.csv`/`test.csv` not found |
| `4` | `EmptyDataError` | Input dataset is empty or has no remaining rows |
| `5` | `MalformedDataError` | Input file is malformed or unsupported |
| `6` | `MissingColumnError` | Required column is missing from dataset |
| `8` | `ReportWriteError` | Failed to write output or create directory |
| `9` | `TransformError` | Target or feature preparation failed |
| `99` | `EXIT_UNEXPECTED` | Unexpected unhandled error |

---

## License

AgriPak is MIT-licensed, as found in the [LICENSE](../../LICENSE) file.
