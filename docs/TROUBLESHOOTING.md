# Environment Troubleshooting Guide

## 1. Quick Diagnostic

Run this diagnostic script to identify common environment issues on Windows:

```powershell
# Save as diagnose.ps1 and run:
#   powershell -ExecutionPolicy Bypass -File diagnose.ps1

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

Quick one-liner diagnostic:

```powershell
python --version;
python -m pip --version;
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('All packages OK')";
Test-Path "data\raw\diabetic_data.csv"
```

## 2. Common Issues and Solutions

### Issue 1: “Could not find platform independent libraries <prefix>”
**Symptom**

```text
Could not find platform independent libraries <prefix>
Consider setting $PYTHONHOME to <prefix>[:<exec_prefix>]
...
```

**Cause**
Corrupted or partially installed Python (very common with a broken global installation).

**Fix (recommended)**
Do not repair the broken Python. Create a clean Conda environment:

```powershell
# Install Miniconda first (see LOCAL_WINDOWS_SETUP.md)

conda create -n anomaly python=3.11 -y
conda activate anomaly
pip install -r requirements.txt
```

### Issue 2: “No module named pip”
**Symptom**

```text
python -m pip
No module named pip
```

**Cause**
System Python is incomplete or broken.

**Fix A (standalone Python)**

```powershell
python -m ensurepip --upgrade

Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
python get-pip.py

python -m pip --version
```

**Fix B (recommended: Conda)**

```powershell
conda create -n anomaly python=3.11 -y
conda activate anomaly
python -m pip --version
```

### Issue 3: “ModuleNotFoundError: No module named 'X'”
**Symptom**

```text
ModuleNotFoundError: No module named 'pandas'
```

**Cause**
Dependencies not installed, or the wrong Python interpreter is being used.

**Fix**

```powershell
# 1. Verify which Python is used
where python

# 2. Activate the project environment
conda activate anomaly

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify imports
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('Success!')"
```

If a single package is missing:

```powershell
pip install pandas        # or scikit-learn, torch, ...
```

If you see `No module named 'src'`, you are not in the project root. Run:

```powershell
cd "path\to\project\root"
```

### Issue 4: Package Version Conflicts
**Symptom**

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Fix**

```powershell
# Option 1: Force reinstall
pip install --upgrade --force-reinstall -r requirements.txt

# Option 2: Fresh environment (recommended)
conda deactivate
conda env remove -n anomaly
conda create -n anomaly python=3.11 -y
conda activate anomaly
pip install -r requirements.txt
```

### Issue 5: PyTorch Installation Fails or DLL Errors
**Symptoms**

```text
ImportError: DLL load failed while importing _C
```
or
```text
ERROR: Could not find a version that satisfies the requirement torch>=2.0
```

**Causes**
- Missing Visual C++ redistributables on Windows.
- Incompatible Python version.

**Fix**

```powershell
# 1. Install Visual C++ Redistributable:
#    https://aka.ms/vs/17/release/vc_redist.x64.exe

# 2. Verify Python version
python --version   # should be between 3.8 and 3.12

# 3. Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# 4. Test
python -c "import torch; print(torch.__version__)"
```

For CPU-only:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 6: Wrong Python Version
**Symptom**

```text
python --version
Python 3.14.0
```
but you expect the Conda environment version (for example 3.11).

**Fix**

```powershell
conda activate anomaly
python --version
where python
```

If activation does not work:

```powershell
conda init powershell
# Close and reopen PowerShell, then:
conda activate anomaly
```

### Issue 7: Jupyter Notebook Kernel Uses Wrong Python
**Symptoms**
- Packages import correctly in terminal but not in notebooks.
- `ModuleNotFoundError` inside Jupyter.

**Fix**

```powershell
conda activate anomaly
pip install jupyter ipykernel
python -m ipykernel install --user --name anomaly --display-name "Python (anomaly)"
jupyter notebook
```

In Jupyter:
Kernel → Change kernel → Python (anomaly), then in a cell:

```python
import sys
print(sys.executable)
```
Path should point to the anomaly environment.

### Issue 8: Data File Not Found
**Symptom**

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/diabetic_data.csv'
```

**Fix**

```powershell
# 1. Check current directory
pwd

# 2. Ensure data directory exists
Test-Path "data\raw"             # if False:
New-Item -ItemType Directory -Path "data\raw" -Force

# 3. Check file
Test-Path "data\raw\diabetic_data.csv"

# 4. If missing, download from the UCI repository and place it there.
```

Optionally check file size:

```powershell
(Get-Item "data\raw\diabetic_data.csv").Length / 1MB
```

### Issue 9: Permission Denied When Writing Results
**Symptom**

```text
PermissionError: [Errno 13] Permission denied: 'results/figures/plot.png'
```

**Fix**

```powershell
# 1. Close viewers (image preview, Excel, etc.)

# 2. Ensure directories exist
New-Item -ItemType Directory -Path "results\figures" -Force
New-Item -ItemType Directory -Path "results\models" -Force
New-Item -ItemType Directory -Path "results\reports" -Force

# 3. If needed, run PowerShell as Administrator
```

### Issue 10: conda Command Not Found
**Symptom**

```text
conda : The term 'conda' is not recognized...
```

**Fix**

```powershell
# 1. Check if Miniconda exists
Test-Path "$env:USERPROFILE\miniconda3"

# 2. Initialize for PowerShell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell

# 3. Close and reopen PowerShell
conda --version
```

If Miniconda is not installed, reinstall it and enable “Add to PATH” during installation.

### Issue 11: PowerShell Script Execution Policy
**Symptom**

```text
File ...\activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

**Fix**

```powershell
# Current session only
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Or for current user (persistent)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Alternatively, use cmd.exe or the Anaconda Prompt.

## 3. Quick Reference Table

| Symptom | Likely Cause | Recommended Fix |
| :--- | :--- | :--- |
| “Could not find platform independent libraries” | Broken Python | Use Miniconda and a clean environment |
| “No module named pip” | Broken Python | Ensurepip or switch to Conda |
| “No module named 'pandas'” | Packages missing | `pip install -r requirements.txt` |
| PyTorch DLL errors | Missing VC++ / Python | Install VC++ and reinstall PyTorch |
| `conda` not recognized | PATH / init issue | `conda init powershell` |
| Wrong Python version | Env not activated | `conda activate anomaly` |
| Data file missing | File not in `data/` | Place CSV under `data/raw/` |
| Notebook cannot import modules | Wrong kernel | Register and select Conda kernel |

## 4. Full Reset (Emergency Recovery)
If the environment is badly broken and you want a clean start:

```powershell
# 1. Backup results (optional)
Copy-Item "results\" -Destination "C:\Backup\results_backup\" -Recurse -Force

# 2. Remove old environment
conda env remove -n anomaly

# 3. (Optional) Reinstall Miniconda if Conda itself is broken

# 4. Recreate environment
conda create -n anomaly python=3.11 -y
conda activate anomaly
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify
python -c "import pandas, numpy, sklearn, torch, matplotlib, seaborn; print('Environment rebuilt successfully.')"
```

If environment problems persist on Windows, you can also run the project in Google Colab or inside a Linux environment (WSL2 or Docker).
