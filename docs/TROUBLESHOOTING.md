# Environment Troubleshooting Guide

## 🎯 Quick Diagnostic

Run this diagnostic script to identify your environment issue:

```powershell
# Save this as diagnose.ps1 and run: powershell -ExecutionPolicy Bypass -File diagnose.ps1

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Environment Diagnostic Tool" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Check 1: Python availability
Write-Host "`n[1] Checking Python installations..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  python: $pythonVersion" -ForegroundColor Green
    $pythonPath = (Get-Command python).Source
    Write-Host "  location: $pythonPath" -ForegroundColor Green
} catch {
    Write-Host "  python: NOT FOUND" -ForegroundColor Red
}

# Check 2: py launcher
Write-Host "`n[2] Checking py launcher..." -ForegroundColor Yellow
try {
    py -0p 2>&1 | ForEach-Object { Write-Host "  $_" }
} catch {
    Write-Host "  py launcher: NOT AVAILABLE" -ForegroundColor Red
}

# Check 3: Conda
Write-Host "`n[3] Checking conda..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    Write-Host "  conda: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "  conda: NOT INSTALLED" -ForegroundColor Red
}

# Check 4: pip
Write-Host "`n[4] Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = python -m pip --version 2>&1
    Write-Host "  pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  pip: NOT AVAILABLE (Python may be broken)" -ForegroundColor Red
}

# Check 5: Required packages
Write-Host "`n[5] Checking required packages..." -ForegroundColor Yellow
$packages = @("pandas", "numpy", "sklearn", "torch", "matplotlib", "seaborn")
foreach ($pkg in $packages) {
    try {
        python -c "import $pkg; print('$pkg OK')" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  $pkg : INSTALLED" -ForegroundColor Green
        } else {
            Write-Host "  $pkg : MISSING" -ForegroundColor Red
        }
    } catch {
        Write-Host "  $pkg : ERROR CHECKING" -ForegroundColor Red
    }
}

# Check 6: Data file
Write-Host "`n[6] Checking data file..." -ForegroundColor Yellow
$dataPath = "data\raw\diabetic_data.csv"
if (Test-Path $dataPath) {
    $size = (Get-Item $dataPath).Length / 1MB
    Write-Host "  diabetic_data.csv: FOUND ($("{0:N2}" -f $size) MB)" -ForegroundColor Green
} else {
    Write-Host "  diabetic_data.csv: NOT FOUND" -ForegroundColor Red
}

Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host "Diagnostic Complete" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
```

**Quick one-liner diagnostic**:
```powershell
python --version; python -m pip --version; python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('All packages OK')"; Test-Path "data\raw\diabetic_data.csv"
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "Could not find platform independent libraries <prefix>"

**Symptoms**:
```
Could not find platform independent libraries <prefix>
Consider setting $PYTHONHOME to <prefix>[:<exec_prefix>]
Python path configuration:
  PYTHONHOME = (not set)
  ...
```

**Diagnosis**: Python installation is corrupted (common with Python 3.14 on Windows)

**Solution**: DO NOT attempt to repair. Create a clean environment:

```powershell
# Recommended: Use Miniconda
# Follow LOCAL_WINDOWS_SETUP.md → "Step-by-Step Installation"

# Quick fix:
# 1. Install Miniconda
# 2. conda create -n anomaly python=3.11 -y
# 3. conda activate anomaly
# 4. pip install -r requirements.txt
```

---

### Issue 2: "No module named pip"

**Symptoms**:
```
python -m pip
No module named pip
```

**Diagnosis**: pip not installed (broken Python installation)

**Solution A** (if using standalone Python):
```powershell
# Try to bootstrap pip
python -m ensurepip --upgrade

# If that fails, download get-pip.py
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
python get-pip.py

# Verify
python -m pip --version
```

**Solution B** (recommended - switch to conda):
```powershell
# Conda includes pip by default
conda create -n anomaly python=3.11 -y
conda activate anomaly
python -m pip --version  # Should work immediately
```

---

### Issue 3: "ModuleNotFoundError: No module named 'X'"

**Symptoms**:
```
python run_pipeline_direct.py
ModuleNotFoundError: No module named 'pandas'
```

**Diagnosis**: Packages not installed OR wrong Python environment

**Solution**:

```powershell
# Step 1: Verify which Python you're using
where python
# Should show path to your conda environment, e.g.:
# C:\Users\Asus\miniconda3\envs\anomaly\python.exe

# Step 2: If wrong Python, activate environment
conda activate anomaly

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Verify imports
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('Success!')"
```

**Common variations**:
- `No module named 'sklearn'` → Install: `pip install scikit-learn`
- `No module named 'torch'` → Install: `pip install torch`
- `No module named 'src'` → Wrong working directory, run: `cd "path\to\project\root"`

---

### Issue 4: Package Version Conflicts

**Symptoms**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
This behaviour is the source of the following dependency conflicts.
```

**Diagnosis**: Version conflicts between packages

**Solution**:

```powershell
# Option 1: Force reinstall with --upgrade
pip install --upgrade --force-reinstall -r requirements.txt

# Option 2: Create fresh environment (recommended)
conda deactivate
conda env remove -n anomaly
conda create -n anomaly python=3.11 -y
conda activate anomaly
pip install -r requirements.txt

# Option 3: Update requirements.txt with specific versions
# Edit requirements.txt to remove version constraints (>= instead of ==)
```

---

### Issue 5: PyTorch Installation Fails or DLL Errors

**Symptoms**:
```
ImportError: DLL load failed while importing _C: The specified module could not be found.
```
OR
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0
```

**Diagnosis**: 
- Missing Visual C++ Redistributables (Windows)
- OR incompatible Python version (3.13+)

**Solution**:

```powershell
# Step 1: Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
# Run and install

# Step 2: Verify Python version
python --version
# Should be 3.8-3.12 (NOT 3.13+)

# Step 3: Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# Step 4: Test import
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# If still fails: Use CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Issue 6: Wrong Python Version Used

**Symptoms**:
```
python --version
Python 3.14.0
```
But you want to use Python 3.11 from conda environment

**Diagnosis**: conda environment not activated OR system Python in PATH before conda

**Solution**:

```powershell
# Step 1: Activate conda environment
conda activate anomaly

# Step 2: Verify Python
python --version  # Should now show 3.11.x

# Step 3: Check which Python is being used
where python
# First result should be conda environment, e.g.:
# C:\Users\Asus\miniconda3\envs\anomaly\python.exe
# If C:\Python314\python.exe appears first, conda environment failed to activate

# Step 4: If activation fails, initialize conda for PowerShell
conda init powershell
# Close and reopen PowerShell, then retry
conda activate anomaly
```

---

### Issue 7: Jupyter Notebook Kernel Issues

**Symptoms**:
- Notebooks can't find installed packages
- "No module named 'pandas'" in Jupyter but works in terminal

**Diagnosis**: Jupyter using wrong Python kernel

**Solution**:

```powershell
# Step 1: Activate your conda environment
conda activate anomaly

# Step 2: Install Jupyter in the environment
pip install jupyter ipykernel

# Step 3: Register kernel
python -m ipykernel install --user --name anomaly --display-name "Python (anomaly)"

# Step 4: Launch Jupyter
jupyter notebook

# Step 5: In notebook, select kernel:
# Kernel → Change kernel → Python (anomaly)

# Step 6: Verify
# Run in notebook cell:
# import sys
# print(sys.executable)
# Should show path to your conda environment
```

---

### Issue 8: Data File Not Found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/diabetic_data.csv'
```

**Diagnosis**: 
- Data file not downloaded
- OR wrong working directory
- OR incorrect file path

**Solution**:

```powershell
# Step 1: Verify current directory
pwd
# Should end with: ...\Ontology-aware Anomaly Detection Toy Pipeline

# Step 2: Check if data directory exists
Test-Path "data\raw"
# If False, create it:
New-Item -ItemType Directory -Path "data\raw" -Force

# Step 3: Check if data file exists
Test-Path "data\raw\diabetic_data.csv"
# If False, you need to download it

# Step 4: Download data (if needed)
# Option A: Download from UCI repository
# Visit: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
# Download diabetic_data.csv

# Option B: If you have it elsewhere, copy it
# Copy-Item "C:\path\to\diabetic_data.csv" -Destination "data\raw\"

# Step 5: Verify file size (should be ~10-20 MB)
(Get-Item "data\raw\diabetic_data.csv").Length / 1MB
```

---

### Issue 9: Permission Denied Errors

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: 'results/figures/plot.png'
```

**Diagnosis**: 
- File is open in another program
- OR insufficient permissions
- OR directory doesn't exist

**Solution**:

```powershell
# Step 1: Close any programs viewing the files (image viewers, Excel, etc.)

# Step 2: Ensure results directories exist
New-Item -ItemType Directory -Path "results\figures" -Force
New-Item -ItemType Directory -Path "results\models" -Force
New-Item -ItemType Directory -Path "results\reports" -Force

# Step 3: Check permissions
icacls "results\"
# You should have (F) Full control

# Step 4: If permission issues, run PowerShell as Administrator
# Right-click PowerShell → Run as Administrator
# Navigate to project and try again

# Step 5: Delete locked files (if safe to do so)
Remove-Item "results\figures\*" -Force
# Then rerun pipeline
```

---

### Issue 10: Conda Command Not Found

**Symptoms**:
```
conda : The term 'conda' is not recognized...
```

**Diagnosis**: 
- Miniconda not installed
- OR not in PATH
- OR PowerShell needs refresh

**Solution**:

```powershell
# Step 1: Check if Miniconda is installed
Test-Path "$env:USERPROFILE\miniconda3"
# OR
Test-Path "C:\ProgramData\Miniconda3"

# Step 2: If installed but not in PATH, manually add
# Find conda location
Get-ChildItem -Path C:\ -Filter "conda.exe" -Recurse -ErrorAction SilentlyContinue

# Step 3: If not installed, install Miniconda
# Download from: https://docs.conda.io/en/latest/miniconda.html
# Run installer, check "Add to PATH"

# Step 4: If installed, initialize PowerShell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
# Close and reopen PowerShell

# Step 5: Verify
conda --version
```

---

### Issue 11: Script Execution Policy Error (PowerShell)

**Symptoms**:
```
.\activate : File ...\activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

**Diagnosis**: PowerShell execution policy blocking conda activation

**Solution**:

```powershell
# Option 1: Bypass for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Option 2: Set for current user (permanent)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Option 3: Use Command Prompt instead of PowerShell
# Open cmd.exe and use:
conda activate anomaly

# Option 4: Use Anaconda Prompt
# Search for "Anaconda Prompt" in Windows Start menu
```

---

## 🔍 Environment Comparison Table

| Symptom | Likely Cause | Recommended Solution | Quick Fix |
|---------|--------------|----------------------|-----------|
| `Could not find platform independent libraries` | Broken Python | Use Miniconda | See Issue 1 |
| `No module named pip` | Broken Python | Use Miniconda | `python -m ensurepip` |
| `No module named 'pandas'` | Packages not installed | `pip install -r requirements.txt` | `pip install pandas` |
| `DLL load failed` (PyTorch) | Missing Visual C++ | Install VC++ Redistributable | Use CPU PyTorch |
| `conda not found` | Not in PATH | `conda init powershell` | Use Anaconda Prompt |
| Wrong Python version | Environment not activated | `conda activate anomaly` | Check `where python` |
| Data file not found | File missing | Download or move file | Check with `Test-Path` |
| Jupyter kernel issues | Wrong kernel | Install ipykernel in env | Change kernel in notebook |

---

## 🚑 Emergency Recovery

If everything fails and you want to start completely fresh:

```powershell
# Step 1: Backup any important files
Copy-Item "results\" -Destination "C:\Backup\results_backup\" -Recurse -Force

# Step 2: Remove all conda environments
conda env remove -n anomaly
# Repeat for any other environments you created

# Step 3: Reinstall Miniconda (optional - only if conda is broken)
# Uninstall Miniconda from Windows Settings → Apps
# Download fresh installer from https://docs.conda.io/en/latest/miniconda.html

# Step 4: Create new environment from scratch
conda create -n anomaly python=3.11 -y
conda activate anomaly
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Verify
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('✅ Environment rebuilt successfully!')"
```

---

## 📞 Still Stuck?

If none of these solutions work:

1. **Run the diagnostic script** at the top of this document
2. **Capture the full error message** (copy entire traceback)
3. **Note your environment**:
   - Windows version: `systeminfo | findstr /B /C:"OS Name" /C:"OS Version"`
   - Python version: `python --version`
   - Conda version (if installed): `conda --version`
4. **Alternative**: Use Google Colab (see `GOOGLE_COLAB_SETUP.md`) - requires zero local setup

**Common last resorts**:
- Switch to Google Colab (fastest)
- Use Windows Subsystem for Linux (WSL2) with conda
- Use Docker container (advanced)
