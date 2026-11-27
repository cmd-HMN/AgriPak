## Preprocessing & Feature Engineering

- **Rows / Columns:** `50,000` rows, `10` raw columns -> `14` processed columns
- **Primary Driver:** `low_yield_risk` (inferred risk floor threshold: `0.00`)
- **Binary Signal Columns (3):** `fertilizer_used, irrigation_used, low_yield_risk`
- **Engineered Features (4):** `binary_signal_sum, driver_deficit, deficit_x_zero_signal, driver_x_signal_sum`

### Engineered Feature Definitions

1. `binary_signal_sum`: Row-wise sum of active binary risk flags (`3` columns), capturing cumulative risk.
2. `driver_deficit`: `max(0, 0.00 - low_yield_risk)`, isolating shortfall below the positive-class median.
3. `deficit_x_zero_signal`: Interaction `driver_deficit * I(binary_signal_sum == 0)`, capturing silent-risk cases.
4. `driver_x_signal_sum`: Interaction `low_yield_risk * binary_signal_sum`, scaling driver magnitude by active flag count.

## Feature Selection & Transformation Summary

- **Selected Features (6):** `fertilizer_used, low_yield_risk, binary_signal_sum, driver_deficit, deficit_x_zero_signal, driver_x_signal_sum`
- **Dropped Noise Features (8):** `rainfall_mm, temperature_c, days_to_harvest, irrigation_used, region, soil_type, crop, weather_condition`
- **Encoding (`onehot`):** `0` categorical -> `0` encoded columns
- **Scaling (`standard`):** `4` numeric columns (`2` zero-variance)

### Selection Manifest

| feature | decision | group | signal_metric | signal_value | reason |
| --- | --- | --- | --- | --- | --- |
| rainfall_mm | drop | numeric | spearman | -0.000 | noise: \|spearman\|=0.0001 < 0.03 |
| temperature_c | drop | numeric | spearman | -0.005 | noise: \|spearman\|=0.0050 < 0.03 |
| days_to_harvest | drop | numeric | spearman | -0.003 | noise: \|spearman\|=0.0029 < 0.03 |
| fertilizer_used | keep | binary | spearman | 1.000 | \|spearman\|=1.0000 >= 0.03 |
| irrigation_used | drop | binary | spearman | 0.005 | noise: \|spearman\|=0.0048 < 0.03 |
| low_yield_risk | keep | binary | spearman | -0.334 | \|spearman\|=0.3342 >= 0.03 |
| region | drop | categorical | cramers_v | 0.015 | noise: cramers_v=0.0147 < 0.05 |
| soil_type | drop | categorical | cramers_v | 0.005 | noise: cramers_v=0.0055 < 0.05 |
| crop | drop | categorical | cramers_v | 0.010 | noise: cramers_v=0.0096 < 0.05 |
| weather_condition | drop | categorical | cramers_v | 0.006 | noise: cramers_v=0.0059 < 0.05 |
| binary_signal_sum | keep | engineered | spearman | 0.516 | domain_engineered_feature |
| driver_deficit | keep | engineered | spearman |  | domain_engineered_feature |

## 5-Fold Stratified CV & Ablation

- **Primary Driver Ablation Delta (HGB):** `+0.0000` ROC-AUC
- **Noise Removal Lift (HGB):** `+0.0000` ROC-AUC

### Cross-Validation Results

| feature_set | n_features | logreg_roc_auc | logreg_std | hgb_roc_auc | hgb_std |
| --- | --- | --- | --- | --- | --- |
| raw_signal | 2 | 1.000 | 0.000 | 1.000 | 0.000 |
| engineered_only | 4 | 0.884 | 0.003 | 0.884 | 0.003 |
| raw_plus_engineered | 6 | 1.000 | 0.000 | 1.000 | 0.000 |
| all_raw_features | 10 | 1.000 | 0.000 | 1.000 | 0.000 |
| ablation_no_primary_driver | 5 | 1.000 | 0.000 | 1.000 | 0.000 |

---

Full preprocessing documentation: [`src/preprocess/README.md`](../preprocess/README.md)
