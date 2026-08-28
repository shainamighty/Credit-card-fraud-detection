# Credit Card Fraud Detection

## Overview

Binary classification of credit card fraud on the classic Kaggle Credit Card
Fraud dataset — 284,807 transactions, 492 frauds (0.17% — extreme class
imbalance). Features `V1-V28` are PCA-anonymized for confidentiality;
`Time` and `Amount` are the only raw, interpretable columns.

## EDA

- No missing values across all 284,807 rows.
- Fraud transactions average ₹122 vs ₹88 for legitimate ones, but the
  median tells a different story — fraud median amount (₹9.25) is actually
  *lower* than legit median (₹22), meaning fraud amount is bimodal: lots of
  very small "card testing" transactions plus a few large ones pull the
  mean up.
- Correlation with `Class` shows `V17`, `V14`, `V12`, `V10` as the strongest
  negative predictors and `V4`, `V11` as the strongest positive predictors —
  these anonymized components were engineered by the original dataset
  authors from real transaction features, so despite not knowing what they
  represent literally, they clearly encode fraud-relevant structure.

## Preprocessing

- `Amount` and `Time` scaled with `StandardScaler` (V1-V28 already
  PCA-scaled, left untouched)
- Stratified 80/20 train/test split — essential given only 492 fraud rows
  total; a non-stratified split risks a test set with almost no fraud
  examples

## Models compared

| Model | PR-AUC | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Train time |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.876** | 0.978 | 0.872 | 0.837 | 0.854 | 7.4s |
| Random Forest | 0.814 | 0.978 | 0.792 | 0.816 | 0.804 | 197s |
| Logistic Regression | 0.719 | 0.972 | 0.061 | 0.918 | 0.114 | 1.2s |

XGBoost is the clear winner on PR-AUC — the metric that matters here, since
ROC-AUC is misleadingly high for all three models due to the extreme class
imbalance (a model can score well on ROC-AUC while still being weak on the
rare positive class).

Logistic Regression with `class_weight='balanced'` is a good example of why
accuracy/ROC-AUC alone are misleading: it catches 92% of fraud (highest
recall) but at only 6% precision — it flags huge numbers of legitimate
transactions as fraud to get there. Not deployable as-is.

## Threshold exploration (XGBoost)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.856 | 0.847 | 0.851 |
| 0.5 | 0.872 | 0.837 | 0.854 |
| 0.7 | 0.882 | 0.837 | 0.859 |
| 0.9 | 0.919 | 0.806 | 0.859 |

Threshold 0.5-0.7 balances precision and recall well without the sharp
recall drop-off seen at 0.9. In a real deployment, the choice would depend
on the relative cost of a missed fraud vs. a false alarm — a lower threshold
if missing fraud is costlier, higher if customer friction from false flags
is the bigger concern.

## Feature importance (XGBoost)

```
V14              0.546
V4               0.076
V12              0.041
V8               0.024
V10              0.024
Amount_scaled    0.021
V7               0.021
V17              0.020
V20              0.018
V26              0.016
```

`V14` dominates by a wide margin — consistent with it showing the second
strongest raw correlation with `Class` in EDA. `V4`, `V12`, `V10`, `V17` also
appear in both the correlation ranking and the model's importance ranking,
cross-validating that these aren't spurious — the model is genuinely relying
on the same signal the EDA surfaced.

## Files

- `eda.py` — data loading, class balance check, amount/correlation analysis
- `train.py` — preprocessing, model training, threshold exploration, feature importance
- `model_comparison_results.csv` — output metrics table
- `data/creditcard.csv.gz` — gzip-compressed dataset. Run `gunzip data/creditcard.csv.gz` (or `gunzip -k data/creditcard.csv.gz` to keep the .gz copy) before running `train.py` or `eda.py`, since both expect `data/creditcard.csv`.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost

## Author

**Shaina Srujitha**
