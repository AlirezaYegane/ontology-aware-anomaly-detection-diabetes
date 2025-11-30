# Local Windows Environment Setup Guide

## 📋 Project Overview

This **Ontology-aware Anomaly Detection Toy Pipeline** is a complete ML system that:

- **Loads & preprocesses** the diabetes hospital readmission dataset (`diabetic_data.csv` - 101,766 records)
- **Trains two anomaly detection models**:
  - **Isolation Forest**: Tree-based unsupervised detector
  - **Autoencoder**: PyTorch neural network trained on normal samples
- **Applies ontology rules**: Clinical domain knowledge penalties for high-risk patients
- **Evaluates performance**: ROC-AUC, PR-AUC, precision/recall curves
- **Generates outputs**: Figures (`.png`), models (`.pkl`), reports (`.csv`)

### What You Need to Know

**Entry Points**:
- `run_pipeline_direct.py` - **Recommended**: Runs entire pipeline without Jupyter
- `run_pipeline.py` - Executes notebooks sequentially (requires Jupyter installed)

**Source Modules** (`src/`):
- `preprocessing.py` - Data loading, cleaning, feature engineering
- `models.py` - IsolationForest and Autoencoder classes
- `ontology.py` - Clinical rule-based penalty computation
- `evaluation.py` - Metrics calculation and visualization
- `utils.py` - Path handling for hybrid local/Colab environments

**Expected Outputs** (after successful run):
- `data/processed/` - Preprocessed features, target, pipeline
- `results/figures/` - ROC/PR curves, score distributions
- `results/models/` - Trained model files
- `results/reports/` - Metrics CSVs

---

## ⚠️ Current Environment Problem

Your existing Python 3.14 at `C:\Python314\python.exe` is **broken**:
- ❌ "Could not find platform independent libraries <prefix>"
- ❌ No `pip` module
- ❌ No packages installed

**DO NOT ATTEMPT TO USE THIS PYTHON**. Follow the recommended solution below.

---

## ✅ RECOMMENDED SOLUTION: Miniconda + Python 3.11

This is the **cleanest, most maintainable, and beginner-friendly** approach for Windows users.

### Why Miniconda?

- ✅ **Isolated environments**: Never pollutes your system Python
- ✅ **Easy to manage**: Simple commands to create/delete environments
- ✅ **Cross-platform**: Same workflow on Windows, macOS, Linux
- ✅ **Package management**: `conda` + `pip` both work seamlessly
- ✅ **Reproducible**: Easy to recreate if something breaks

---

## 🚀 Step-by-Step Installation (Recommended Path)

### Step 1: Install Miniconda

1. **Download Miniconda**:
   - Go to: https://docs.conda.io/en/latest/miniconda.html
   - Download **Miniconda3 Windows 64-bit** (latest version)
   - File will be named something like `Miniconda3-latest-Windows-x86_64.exe`

2. **Run the installer**:
   - Double-click the downloaded `.exe` file
   - Click "Next" → "I Agree"
   - **Important**: Select "Just Me (recommended)"
   - **Important**: Check ✅ "Add Miniconda3 to my PATH environment variable" (even though it says not recommended, this makes life easier)
   - Click "Install"
   - Wait for installation to complete (~5 minutes)

3. **Verify installation**:
   - Open a **new** PowerShell window (important: must be new, or run `refreshenv` if using chocolatey)
   - Run:
     ```powershell
     conda --version
     ```
   - You should see: `conda 24.x.x` or similar

### Step 2: Create Clean Python 3.11 Environment

Open PowerShell and run these commands **one by one**:

```powershell
# Navigate to your project folder
cd "C:\Users\Asus\Desktop\PY projects\Ontology-aware Anomaly Detection Toy Pipeline"

# Create a new conda environment with Python 3.11
conda create -n anomaly python=3.11 -y

# Activate the environment
conda activate anomaly
```

**What you should see**:
- After `conda create`: Progress bar → "Preparing transaction: done" → "Verifying transaction: done"
- After `conda activate`: Your prompt changes to `(anomaly) PS C:\Users\Asus\...`

**Troubleshooting**:
- If `conda activate` doesn't work, run: `conda init powershell` and restart PowerShell
- If you see "conda not found", restart PowerShell or add Miniconda to PATH manually

### Step 3: Install Project Dependencies

**Still in the same PowerShell window** (with `(anomaly)` prefix visible):

```powershell
# Verify you're in the right environment
python --version
# Should output: Python 3.11.x

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

**What you should see**:
- `python --version` → `Python 3.11.9` (or similar)
- `pip install` → Downloads pandas, numpy, scikit-learn, torch, matplotlib, seaborn
- Installation takes ~3-5 minutes (PyTorch is large)

**Expected output**:
```
Successfully installed pandas-2.x.x numpy-1.26.x scikit-learn-1.4.x torch-2.x.x matplotlib-3.8.x seaborn-0.13.x ...
```

### Step 4: Verify Installation

**Quick verification** (still in PowerShell with `(anomaly)` active):

```powershell
# Test all imports
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('✅ All packages imported successfully!')"
```

**If successful, you'll see**:
```
✅ All packages imported successfully!
```

**If you see an error**:
- `ModuleNotFoundError: No module named 'pandas'` → Re-run `pip install -r requirements.txt`
- `DLL load failed` with torch → Install Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Step 5: Ensure Data File Exists

```powershell
# Check if data file exists
Test-Path "data\raw\diabetic_data.csv"
```

**If output is `False`**:
- Download `diabetic_data.csv` from the UCI repository or your data source
- Place it in `data\raw\` folder
- Verify again with the command above (should return `True`)

### Step 6: Run the Pipeline

```powershell
# Option 1: Direct execution (no Jupyter needed)
python run_pipeline_direct.py

# Option 2: Execute notebooks (requires Jupyter)
pip install jupyter  # If not already installed
python run_pipeline.py
```

**Expected runtime**: ~3-5 minutes

**What you should see**:
```
═══════════════════════════════════════════════
Ontology-Aware Anomaly Detection Pipeline
═══════════════════════════════════════════════

STEP 1/4: Data Loading & Preprocessing
✅ Loaded 101766 records
✅ Created binary target: 10842 readmitted <30 days (10.6%)
✅ Feature matrix shape: (98053, 38)
✅ Saved to data/processed/

STEP 2/4: Isolation Forest Training
✅ Trained on 78442 normal samples
✅ ROC-AUC: 0.64
✅ PR-AUC: 0.15

STEP 3/4: Autoencoder Training
[Epoch 1/20] Loss: 0.4523
[Epoch 20/20] Loss: 0.1234
✅ Model trained

STEP 4/4: Ontology Evaluation
✅ Applied clinical rules
✅ Combined scores (70% ML + 30% Ontology)
✅ Final ROC-AUC: 0.67 (+0.03 improvement)

═══════════════════════════════════════════════
Pipeline complete! Check results/ folder
═══════════════════════════════════════════════
```

**Generated files**:
- `data/processed/X_features.csv`, `y_target.csv`, `preprocessor.pkl`
- `results/figures/if_roc_pr_curves.png`, `autoencoder_roc_pr_curves.png`, `ontology_comparison_curves.png`
- `results/reports/metrics_summary.csv`

---

## 🔧 Common Errors & Solutions

### Error 1: `conda: command not found`

**Cause**: Miniconda not in PATH

**Solution**:
1. Restart PowerShell
2. OR manually add to PATH:
   - Search Windows for "Environment Variables"
   - Edit "Path" → New → Add `C:\Users\Asus\miniconda3\Scripts` and `C:\Users\Asus\miniconda3`
3. OR use Anaconda Prompt instead of PowerShell (search for it in Start menu)

### Error 2: `ModuleNotFoundError: No module named 'pandas'`

**Cause**: Packages not installed OR wrong Python environment active

**Solution**:
```powershell
# Check which Python you're using
where python
# Should show: C:\Users\Asus\miniconda3\envs\anomaly\python.exe

# If not, activate environment
conda activate anomaly

# Reinstall packages
pip install -r requirements.txt
```

### Error 3: `FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/diabetic_data.csv'`

**Cause**: Data file not placed in correct location

**Solution**:
1. Download `diabetic_data.csv`
2. Create folder: `mkdir -p data\raw` (if it doesn't exist)
3. Move file to: `data\raw\diabetic_data.csv`
4. Verify: `Test-Path "data\raw\diabetic_data.csv"` → should return `True`

### Error 4: PyTorch installation fails or `DLL load failed`

**Cause**: Missing Visual C++ redistributables (Windows)

**Solution**:
1. Download Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install it
3. Reinstall PyTorch: `pip install torch --force-reinstall`

### Error 5: `Could not find platform independent libraries`

**Cause**: You're still using the broken Python 3.14

**Solution**:
```powershell
# Check which Python is running
python --version
where python

# If it shows C:\Python314\python.exe, you forgot to activate conda environment
conda activate anomaly

# Verify now
python --version  # Should show Python 3.11.x
```

### Error 6: Environment activation fails

**Cause**: PowerShell not initialized for conda

**Solution**:
```powershell
# Initialize conda for PowerShell
conda init powershell

# Close and reopen PowerShell, then try again
conda activate anomaly
```

---

## 🔄 Deactivating and Reusing the Environment

**To deactivate** (exit the conda environment):
```powershell
conda deactivate
```

**To reactivate later**:
```powershell
cd "C:\Users\Asus\Desktop\PY projects\Ontology-aware Anomaly Detection Toy Pipeline"
conda activate anomaly
python run_pipeline_direct.py
```

**To delete the environment** (if you want to start fresh):
```powershell
conda deactivate
conda env remove -n anomaly
# Then recreate from Step 2
```

---

## ⚙️ Alternative Solutions (Optional - NOT Recommended)

### Option A: Fix Existing Python 3.14

> ⚠️ **Not recommended** - Python 3.14 may have compatibility issues with PyTorch

```powershell
cd "C:\Python314"
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r "C:\Users\Asus\Desktop\PY projects\Ontology-aware Anomaly Detection Toy Pipeline\requirements.txt"
```

### Option B: Install New Standalone Python 3.11

1. Download Python 3.11 from https://www.python.org/downloads/
2. During installation, check ✅ "Add Python 3.11 to PATH"
3. Open new PowerShell:
   ```powershell
   python --version  # Should show 3.11.x
   pip install -r requirements.txt
   ```

---

## ✅ Success Checklist

Before running the pipeline, verify:

- [ ] `conda --version` works
- [ ] `conda activate anomaly` changes prompt to `(anomaly) PS ...`
- [ ] `python --version` shows `Python 3.11.x`
- [ ] `python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn"` runs without error
- [ ] `Test-Path "data\raw\diabetic_data.csv"` returns `True`
- [ ] You are in the project root directory

If all boxes checked → **You're ready to run the pipeline!** 🚀

---

## 📞 Need Help?

If you encounter issues not covered here:

1. Check `requirements.txt` for exact package versions
2. Try creating a fresh conda environment
3. Verify your Windows version supports Python 3.11
4. Consider using Google Colab (see `GOOGLE_COLAB_SETUP.md`) as a zero-setup alternative

**Remember**: Miniconda environments are disposable - if something breaks, just delete and recreate! 
