# Environment Setup & Execution Guide (Summary)

## 1. Project Overview

This repository implements an **ontology-aware anomaly detection pipeline** for predicting early hospital readmission in patients with diabetes, using the public *Diabetes 130-US hospitals* dataset.

The pipeline consists of four main components:

1. **Exploratory Data Analysis (EDA)**
   - Loads `diabetic_data.csv` (≈100k hospital encounters).
   - Explores demographics, lab results, and medication patterns.
   - Quantifies class imbalance (≈10–11% readmitted within 30 days).

2. **Baseline Anomaly Detection**
   - **Isolation Forest** as an unsupervised baseline.
   - Trained on patients **not** readmitted within 30 days.
   - Produces anomaly scores for all encounters.

3. **Deep Learning Baseline**
   - **Feed-forward autoencoder** implemented in PyTorch.
   - Trained to reconstruct normal patterns.
   - Reconstruction error is used as an anomaly score.
   - Automatically uses GPU if available.

4. **Ontology-Enhanced Evaluation**
   - Encodes simple, transparent clinical rules such as:
     - Poor glycaemic control with no medication change.
     - Very high glucose with short length of stay.
     - Frequent inpatient admissions and emergency visits.
   - Converts rule activations into an ontology penalty in [0, 1].
   - Combines ML scores with clinical penalties and evaluates whether
     the ontology improves ROC-AUC and PR-AUC.

---

## 2. How to Run the Pipeline

### 2.1 Main Entry Point (Recommended)

**Script:** `run_pipeline_direct.py`  
**Usage:**

```bash
python run_pipeline_direct.py
```

This script:

- Loads and preprocesses `data/raw/diabetic_data.csv`.
- Trains all baseline models:
  - Isolation Forest
  - Autoencoder
  - Decision Tree (supervised baseline)
  - Random Forest (supervised baseline)
- Applies the ontology rule layer.
- Saves figures, reports, and logs under `results/`.

It does not require Jupyter and is the recommended way to reproduce the results.

### 2.2 Notebooks (Interactive Exploration)
The following notebooks live in `notebooks/` and are aligned with the current `src/` code:

- **01_eda.ipynb**
  - Exploratory data analysis only (no models, no ontology).

- **02_baseline_if.ipynb**
  - Unified baseline notebook:
    - Isolation Forest and Autoencoder (unsupervised)
    - Decision Tree and Random Forest (supervised)
  - Side-by-side comparison using common metrics and plots.

- **04_ontology_eval.ipynb**
  - Ontology-aware evaluation:
    - Applies clinical rules to the test set.
    - Performs a small grid search over λ values for combining ML scores with ontology penalties.
  - Visualises ROC / PR curves for:
    - ML-only models
    - ML + ontology variants
  - Includes a short case-study section showing individual patients where rules fire and scores change.

The original experimental notebooks have been archived under `archive/notebooks/` to keep the main workflow clean.

## 3. Expected Outputs
After running `run_pipeline_direct.py`, the following directories will be populated:

```text
data/processed/
  X_features.csv          # Preprocessed feature matrix
  y_target.csv            # Binary target (readmitted < 30 days)
  preprocessor.pkl        # Fitted preprocessing pipeline

results/figures/
  *.png                   # ROC/PR curves and related plots

results/models/
  isolation_forest.pkl    # Trained Isolation Forest
  autoencoder.pth         # Trained autoencoder weights (PyTorch)

results/reports/
  data_filtering_summary.json
  data_filtering_summary.md
  multi_split_metrics.json
  multi_split_metrics.md
```

Typical performance (approximate, may vary slightly by random seed):

- **Isolation Forest**: ROC-AUC ≈ 0.63–0.65, PR-AUC slightly above the class-imbalance baseline.
- **Autoencoder**: ROC-AUC often slightly higher than Isolation Forest.
- **Supervised baselines**: higher absolute performance but rely on labels.
- **Ontology layer**: usually adds a modest but consistent improvement in ROC-AUC and PR-AUC for the unsupervised models.

## 4. Setup Options
You can run the project either locally (recommended for repeated experiments) or in a hosted notebook environment.

### 4.1 Local Setup (Windows + Conda)
Install Miniconda.

Create and activate an environment:

```powershell
conda create -n anomaly python=3.11 -y
conda activate anomaly
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify that the raw data exists:

- Place `diabetic_data.csv` under `data/raw/`.

Run the main pipeline:

```powershell
python run_pipeline_direct.py
```

### 4.2 Running in Colab (Optional)
Open a new notebook.

Clone the repository or upload the project.

Install dependencies:

```python
!pip install -r requirements.txt
```

Run:

```python
!python run_pipeline_direct.py
```

The utility functions in `src/utils.py` handle path setup for both local and Colab environments.

## 5. Quick Diagnostics
Before running the full pipeline, you can sanity-check the environment:

```bash
python --version
python -c "import pandas, numpy, sklearn, torch, matplotlib; print('All core packages imported.')"
```

Also verify that:

- `data/raw/diabetic_data.csv` exists.
- `results/` is writable (the script will create subdirectories if needed).

If issues occur (e.g. ModuleNotFoundError, missing data files, or PyTorch installation problems), consult `TROUBLESHOOTING.md` for detailed, case-by-case fixes.

## 6. Project Structure (High-Level)

```text
project-root/
  README.md
  requirements.txt
  run_pipeline_direct.py
  setup_project.py
  src/
    config.py
    preprocessing.py
    models.py
    ontology.py
    evaluation.py
    filter_tracker.py
    logger.py
    utils.py
  notebooks/
    01_eda.ipynb
    02_baseline_if.ipynb
    04_ontology_eval.ipynb
  data/
    raw/
      diabetic_data.csv
    processed/
  results/
    figures/
    models/
    reports/
  tests/
  archive/
    notebooks/
    docs/
    scripts/
```

This guide is intentionally short and implementation-oriented so that a reviewer can install the environment and reproduce the main results with a single command:

```bash
python run_pipeline_direct.py
```
