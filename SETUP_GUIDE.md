# Environment Setup & Execution Guide - Quick Summary

## 📦 What This Project Does

This is an **Ontology-aware Anomaly Detection Pipeline** for predicting early hospital readmissions in diabetes patients.

### Pipeline Components

1. **Exploratory Data Analysis (EDA)**
   - Loads `diabetic_data.csv` (101,766 hospital encounters)
   - Analyzes demographics, lab results, medication patterns
   - Identifies class imbalance (11% readmitted <30 days)

2. **Baseline Anomaly Detection**
   - **Isolation Forest**: Tree-based unsupervised anomaly detector
   - Trained on normal samples (patients NOT readmitted early)
   - Computes anomaly scores for all patients

3. **Deep Learning Approach**
   - **PyTorch Autoencoder**: Neural network trained to reconstruct normal patterns
   - Reconstruction error → anomaly score
   - GPU-accelerated (if available)

4. **Ontology-Enhanced Evaluation**
   - Clinical domain knowledge encoded as rules:
     - High-risk: Poor glycemic control + no medication changes
     - High-risk: Very high glucose + insufficient lab procedures
     - Medium-risk: High medication burden + short hospital stay
   - Combines ML scores with clinical penalties
   - Evaluates if ontology improves prediction (ROC-AUC, PR-AUC)

### Entry Points

- **`run_pipeline_direct.py`** ⭐ **RECOMMENDED**
  - Runs entire pipeline without Jupyter
  - Progress bars, metrics printed to console
  - Saves results to `results/` directory
  
- **`run_pipeline.py`**
  - Executes all notebooks sequentially
  - Requires Jupyter installed
  
- **Individual notebooks** (interactive):
  - `01_eda.ipynb` - Exploratory analysis
  - `02_baseline_if.ipynb` - Isolation Forest
  - `03_autoencoder.ipynb` - PyTorch Autoencoder
  - `04_ontology_eval.ipynb` - Rule-based enhancement

### Expected Outputs

After successful run, you'll see:

```
data/processed/
├── X_features.csv          # Preprocessed features (98K × 38)
├── y_target.csv            # Binary target (readmitted <30 days)
└── preprocessor.pkl        # Scikit-learn pipeline

results/figures/
├── if_roc_pr_curves.png                 # Isolation Forest performance
├── autoencoder_roc_pr_curves.png        # Autoencoder performance
├── ontology_comparison_curves.png       # ML vs ML+Ontology
└── score_distributions.png              # Anomaly score histograms

results/models/
├── isolation_forest.pkl
└── autoencoder.pth

results/reports/
├── if_metrics.csv
├── autoencoder_metrics.csv
└── ontology_comparison.csv
```

**Key metrics** (typical performance):
- ROC-AUC: ~0.64 (Isolation Forest), ~0.65 (Autoencoder)
- PR-AUC: ~0.15 (baseline is 0.11 due to class imbalance)
- Ontology improvement: +0.03-0.05 ROC-AUC

---

## 🚀 Setup Options - Choose Your Path

### Path 1: Local Windows (Miniconda) ⭐ **RECOMMENDED FOR SERIOUS WORK**

**When to use**: Development, repeated experiments, offline work

**Detailed guide**: [`LOCAL_WINDOWS_SETUP.md`](LOCAL_WINDOWS_SETUP.md)

**Quick steps**:
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Create environment:
   ```powershell
   conda create -n anomaly python=3.11 -y
   conda activate anomaly
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run pipeline:
   ```powershell
   python run_pipeline_direct.py
   ```

**Pros**: ✅ Permanent environment, ✅ Works offline, ✅ Full control
**Cons**: ❌ ~10 min setup time

---

### Path 2: Google Colab (Zero Setup) ⭐ **RECOMMENDED FOR QUICK TESTING**

**When to use**: First-time testing, no local Python, need GPU

**Detailed guide**: [`GOOGLE_COLAB_SETUP.md`](GOOGLE_COLAB_SETUP.md)

**Quick steps**:
1. Open https://colab.research.google.com/
2. Upload project zip OR clone from GitHub:
   ```python
   !git clone <your-repo-url>
   %cd "project-folder"
   ```
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Run pipeline:
   ```python
   !python run_pipeline_direct.py
   ```

**Pros**: ✅ 2-minute setup, ✅ Free GPU, ✅ No installation
**Cons**: ❌ Internet required, ❌ Session resets

---

### Path 3: Fix Broken Local Python (Advanced)

**Only if**: You're comfortable with command line troubleshooting

**See**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)

---

## 🐛 Common Issues - Quick Reference

| Error | Quick Fix | Details |
|-------|-----------|---------|
| `Could not find platform independent libraries` | Use Miniconda (don't fix broken Python) | [LOCAL_WINDOWS_SETUP.md](LOCAL_WINDOWS_SETUP.md) |
| `No module named 'pandas'` | `conda activate anomaly` then `pip install -r requirements.txt` | [Issue #3](TROUBLESHOOTING.md#issue-3-modulenotfounderror-no-module-named-x) |
| `FileNotFoundError: diabetic_data.csv` | Download data, place in `data/raw/` | [Issue #8](TROUBLESHOOTING.md#issue-8-data-file-not-found) |
| PyTorch `DLL load failed` | Install VC++ Redistributable: [link](https://aka.ms/vs/17/release/vc_redist.x64.exe) | [Issue #5](TROUBLESHOOTING.md#issue-5-pytorch-installation-fails-or-dll-errors) |
| `conda not found` | Restart PowerShell OR use Anaconda Prompt | [Issue #10](TROUBLESHOOTING.md#issue-10-conda-command-not-found) |
| Jupyter can't find `src` modules | Install kernel: `python -m ipykernel install --user --name anomaly` | [Issue #7](TROUBLESHOOTING.md#issue-7-jupyter-notebook-kernel-issues) |

**Full troubleshooting guide**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📊 Verifying Your Environment

Run this before executing the pipeline:

```powershell
# Windows PowerShell
python --version                    # Should show 3.11.x (if using Miniconda)
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('✅ All packages OK')"
Test-Path "data\raw\diabetic_data.csv"   # Should return True
```

**Expected output**:
```
Python 3.11.9
✅ All packages OK
True
```

If all checks pass → **You're ready to run!** 🚀

---

## 📁 Project Structure Reference

```
Ontology-aware Anomaly Detection Toy Pipeline/
├── LOCAL_WINDOWS_SETUP.md      ← Start here for local installation
├── GOOGLE_COLAB_SETUP.md       ← Start here for Colab
├── TROUBLESHOOTING.md          ← Check here if errors occur
├── README.md                   ← Project overview
├── requirements.txt            ← Dependencies
│
├── run_pipeline_direct.py      ← Main entry point (no Jupyter)
├── run_pipeline.py             ← Notebook-based execution
├── setup_project.py            ← Automated local setup helper
│
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv   ← You must provide this
│   └── processed/              ← Generated by pipeline
│
├── src/                        ← Core modules
│   ├── preprocessing.py        ← Data loading, cleaning, features
│   ├── models.py               ← Isolation Forest & Autoencoder
│   ├── ontology.py             ← Clinical rules
│   ├── evaluation.py           ← Metrics, plots
│   └── utils.py                ← Path handling (Colab/local)
│
├── notebooks/                  ← Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline_if.ipynb
│   ├── 03_autoencoder.ipynb
│   └── 04_ontology_eval.ipynb
│
└── results/                    ← Generated outputs
    ├── figures/                ← PNG plots
    ├── models/                 ← Saved models
    └── reports/                ← CSV metrics
```

---

## 🎯 Decision Matrix: Which Setup Should I Use?

| Your Situation | Recommended Path | Time |
|----------------|------------------|------|
| 🆕 First time trying the project | Google Colab | 2 min |
| 💼 Serious development / research | Local Miniconda | 10 min |
| 🚨 Broken local Python (3.14) | Google Colab OR Miniconda | 2 / 10 min |
| 📴 Need offline access | Local Miniconda | 10 min |
| 🎮 Want GPU for autoencoder | Google Colab (free T4) | 2 min |
| 🔁 Running multiple experiments | Local Miniconda | 10 min (once) |
| 📱 No admin rights on PC | Google Colab | 2 min |

---

## 📞 Need More Help?

1. **For local setup**: Read [`LOCAL_WINDOWS_SETUP.md`](LOCAL_WINDOWS_SETUP.md)
2. **For Colab setup**: Read [`GOOGLE_COLAB_SETUP.md`](GOOGLE_COLAB_SETUP.md)
3. **For errors**: Run diagnostic in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
4. **For project details**: Read [`README.md`](README.md)

---

## ✅ Success Looks Like

After running the pipeline, you should see:

```powershell
═══════════════════════════════════════════════
Ontology-Aware Anomaly Detection Pipeline
═══════════════════════════════════════════════

STEP 1/4: Data Loading & Preprocessing
✅ Loaded 101766 records
✅ Created binary target: 10842 readmitted <30 days (10.6%)
✅ Feature matrix shape: (98053, 38)

STEP 2/4: Isolation Forest Training
✅ Trained on 78442 normal samples
✅ ROC-AUC: 0.64, PR-AUC: 0.15

STEP 3/4: Autoencoder Training
[Epoch 20/20] Loss: 0.1234
✅ ROC-AUC: 0.65, PR-AUC: 0.16

STEP 4/4: Ontology Evaluation
✅ Applied clinical rules
✅ Final ROC-AUC: 0.67 (+0.03 improvement)

═══════════════════════════════════════════════
Pipeline complete! Results saved to results/
═══════════════════════════════════════════════
```

**Outputs**: Check `results/figures/` for visualizations 📊
