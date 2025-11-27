# Data Filtering Summary Report

## Filtering Steps

| Step | Before | After | Removed | % Removed |
|------|-------:|------:|--------:|----------:|
| Load raw dataset (101,766 rows × 50 columns) | 101,766 | 101,766 | 0 | 0.0% |
| Select 14 features (from 50 columns) | 101,766 | 101,766 | 0 | 0.0% |
| Replace '?' with NaN (2,273 cells affected) | 101,766 | 101,766 | 0 | 0.0% |
| Drop rows with missing values in critical columns ['readmitted', 'race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient', 'number_emergency', 'change', 'diabetesMed'] (2,273 rows removed) | 101,766 | 99,493 | 2,273 | 2.2% |
| Create binary target (readmitted <30 days) | 99,493 | 99,493 | 0 | 0.0% |
| Impute missing values (numerical: median, categorical: 'Unknown') | 99,493 | 99,493 | 0 | 0.0% |
| OneHotEncode + StandardScale (14 -> 30 features) | 99,493 | 99,493 | 0 | 0.0% |

## Class Balance

- **Total samples**: 99,493
- **Positive class (y=1)**: 11,169 (11.2%)
- **Negative class (y=0)**: 88,324 (88.8%)

---
*Generated automatically by FilterTracker*
