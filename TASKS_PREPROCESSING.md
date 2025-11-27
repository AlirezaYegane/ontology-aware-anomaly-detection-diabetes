# Data Filtering & Preprocessing Instrumentation Tasks

## Overview
This document tracks the implementation of comprehensive data filtering instrumentation for the preprocessing pipeline.

**Goal**: Understand exactly how the dataset transforms from ~101,766 rows × 50 columns to the final feature matrix, with full traceability and class balance reporting.

---

## ✅ Completed Tasks

### [x] 1. Analyze Current Preprocessing Pipeline
- [x] Identified all filtering/cleaning steps in `src/preprocessing.py`
- [x] Documented each step that reduces row count
- [x] Mapped the complete data transformation flow

**Key findings**:
- Main filtering occurs in `clean_data()` function
- Primary data loss happens at `dropna()` step (removes all rows with ANY missing values)
- Current pipeline: Load → Feature Selection → Replace '?' → Drop NaN → Target Creation → Encode/Scale

---

### [x] 2. Implement Filter Instrumentation

**Created**: `src/filter_tracker.py`
- Centralized `FilterTracker` class for tracking filtering operations
- Tracks: step_name, description, n_before, n_after, n_removed, pct_removed
- Methods: `track_step()`, `track_class_balance()`, `print_summary_table()`, `save_summary()`

**Modified**: `src/preprocessing.py`
- Added `FilterTracker` import and integration
- Instrumented `clean_data()` with step-by-step tracking:
  - Feature selection (50 → 15 columns)
  - Replace '?' with NaN
  - Drop rows with missing values ⚠️ AGGRESSIVE FILTER
- Instrumented `build_feature_matrix()` to track:
  - Initial dataset load
  - All cleaning steps (via clean_data)
  - Target variable creation
  - Feature encoding/scaling
  - Final class balance

**Modified**: `run_pipeline_direct.py`
- Ensured `results/reports/` directory is created for summary files

---

### [x] 3. Build Data Filtering Summary

**Console output**: 
- Formatted table showing each filtering step with before/after counts
- Class balance summary (total samples, positive/negative counts, ratios)

**File outputs** (saved to `results/reports/`):
- `data_filtering_summary.json` - Machine-readable format
- `data_filtering_summary.md` - Human-readable Markdown table

---

### [x] 4. Review Aggressive Cleaning Rules

**Identified aggressive filter**: `dropna()` in `clean_data()`
- Removes ALL rows with ANY missing values
- Potentially removes significant portion of dataset
- May be overly conservative

**Added documentation**:
- ⚠️ AGGRESSIVE FILTER warnings in code comments
- Detailed docstring explaining rationale for each filtering step
- TODO comments suggesting alternative approaches:
  - Population-based imputation
  - Feature-specific imputation strategies
  - Keeping rows with partial data if key features present

**Recommendation**: Analyze which features have the most missing values before deciding on imputation strategy.

---

### [x] 5. Implement Class Balance Summary

**Implemented in** `FilterTracker.track_class_balance()`:
- Counts total samples
- Counts positive class (readmitted <30 days, y=1)
- Counts negative class (not readmitted <30 days, y=0)
- Calculates positive ratio (percentage)

**Outputs**:
- Console: Human-readable summary
- Files: Included in both JSON and Markdown summaries

---

### [x] 6. Create Task Tracking File

**Created**: This file (`TASKS_PREPROCESSING.md`)
- Documents all completed instrumentation work
- Lists filtering steps and their impact
- Identifies aggressive filters
- Notes open questions and future work

---

## 📊 Data Filtering Results

*Will be populated after running the pipeline*

### Filtering Steps Summary

| Step | Description | Rows Before | Rows After | Removed | % Removed |
|------|-------------|------------|------------|---------|----------|
| TBD | TBD | TBD | TBD | TBD | TBD |

### Class Balance

- **Total samples**: TBD
- **Positive class (y=1)**: TBD (TBD%)
- **Negative class (y=0)**: TBD (TBD%)

---

## 🔍 Key Insights

### Most Aggressive Filtering Step(s)
*To be determined after running pipeline*

### Potential Issues
- **Missing value handling**: Current `dropna()` approach may be too aggressive
- **Data loss**: Need to verify how much data is lost and whether it's acceptable
- **Class imbalance**: Need to check if filtering disproportionately affects one class

---

## 📝 Future Work / Open Questions

### [ ] TODO: Analyze Missing Value Patterns
- Which features have the most missing values?
- Is there a pattern to missingness (MCAR, MAR, MNAR)? - Would imputation preserve or improve model performance?

### [ ] TODO: Evaluate Alternative Imputation Strategies
- Mean/median imputation for numerical features
- Mode imputation for categorical features
- KNN imputation
- Model-based imputation (e.g., MICE)

### [ ] TODO: Class Balance Analysis
- Does filtering disproportionately remove positive or negative cases?
- Should we consider stratified sampling before filtering?
- Do we need SMOTE or other resampling techniques?

### [ ] TODO: Feature-Specific Thresholds
- Instead of dropping rows with ANY missing value, could we:
  - Allow missing values in less important features?
  - Set a threshold (e.g., keep rows with >80% complete data)?
  - Create missingness indicator features?

---

## 📁 Files Modified/Created

### Created
- `src/filter_tracker.py` - FilterTracker instrumentation module
- `TASKS_PREPROCESSING.md` - This tracking document

### Modified
- `src/preprocessing.py` - Added comprehensive filtering instrumentation
- `run_pipeline_direct.py` - Ensured reports directory creation

### Generated (after pipeline run)
- `results/reports/data_filtering_summary.json`
- `results/reports/data_filtering_summary.md`

---

*Last updated: 2025-11-27*
*Maintained by: Ontology-aware Anomaly Detection Pipeline Team*
