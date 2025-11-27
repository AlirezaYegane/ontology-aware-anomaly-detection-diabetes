# Google Colab Setup Guide

## 🌐 Zero Local Setup - Run Entirely in the Cloud

Google Colab provides **free GPU/TPU access** and requires **no local Python installation**. Perfect for quick testing or if you're facing local environment issues.

---

## 🚀 Quick Start (3 Options)

### Option 1: GitHub Repository (Recommended if repo is public)

If your project is hosted on GitHub:

**Step 1**: Open Google Colab
- Go to: https://colab.research.google.com/

**Step 2**: Create a new notebook or use an existing one

**Step 3**: Clone the repository (first cell)

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Change to project directory
%cd "YOUR_REPO_NAME"

# Verify structure
!ls -la
```

**Replace**:
- `YOUR_USERNAME` with your GitHub username
- `YOUR_REPO_NAME` with your repository name

**Example**:
```python
!git clone https://github.com/yourusername/ontology-anomaly-detection.git
%cd "ontology-anomaly-detection"
```

---

### Option 2: Upload Project Zip to Colab

If your project is not on GitHub or you prefer local files:

**Step 1**: Create a zip file
- On Windows: Right-click project folder → "Send to" → "Compressed (zipped) folder"
- **Include**: All folders (`data/`, `src/`, `notebooks/`, `results/`) and files (`requirements.txt`, `run_pipeline_direct.py`, etc.)
- **Exclude** (optional): `.git/`, `venv/`, `__pycache__/`, `*.pyc`

**Step 2**: Open Google Colab
- Go to: https://colab.research.google.com/
- Create a new notebook

**Step 3**: Upload and unzip (first cell)

```python
# Upload the zip file
from google.colab import files
print("📤 Select your project zip file...")
uploaded = files.upload()  # Click "Choose Files" and select your .zip

# Get the uploaded filename (usually the first and only file)
zip_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {zip_filename}")

# Unzip
!unzip -q "{zip_filename}"

# Get the extracted folder name (adjust if needed)
# If your zip contains a root folder, use that name
# If not, you might need to adjust the path
import os
folders = [f for f in os.listdir('.') if os.path.isdir(f) and f != 'sample_data']
project_folder = folders[0] if folders else '.'
print(f"📁 Project folder: {project_folder}")

# Change to project directory
if project_folder != '.':
    %cd "{project_folder}"

# Verify structure
!ls -la
```

**Expected output**:
```
data/  notebooks/  results/  src/  requirements.txt  run_pipeline_direct.py  ...
```

---

### Option 3: Upload to Google Drive and Mount

For repeated use, store your project on Google Drive:

**Step 1**: Upload project folder to Google Drive
- Open Google Drive: https://drive.google.com/
- Click "New" → "Folder upload"
- Select your project folder

**Step 2**: In Colab, mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
%cd "/content/drive/MyDrive/YOUR_PROJECT_FOLDER_NAME"

# Verify
!ls -la
```

**Replace** `YOUR_PROJECT_FOLDER_NAME` with actual folder name (e.g., `Ontology-aware Anomaly Detection Toy Pipeline`)

---

## 📦 Installing Dependencies

**After** setting up the project (any option above), install requirements:

```python
# Upgrade pip
!pip install --upgrade pip

# Install project dependencies
!pip install -r requirements.txt
```

**Expected output**:
```
Successfully installed pandas-2.x.x numpy-1.26.x scikit-learn-1.4.x torch-2.x.x ...
```

**Runtime**: ~2-3 minutes

---

## 📊 Ensuring Data is Available

### Option A: Data Already in Repository

If `data/raw/diabetic_data.csv` is included in your git repo or zip:

```python
# Verify data file exists
import os
data_path = "data/raw/diabetic_data.csv"
if os.path.exists(data_path):
    print(f"✅ Data file found: {data_path}")
    print(f"   Size: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")
else:
    print(f"❌ Data file NOT found: {data_path}")
```

### Option B: Data NOT in Repository (Upload Separately)

If you excluded `diabetic_data.csv` from git/zip (large file):

```python
# Upload data file to Colab
from google.colab import files
import os
import shutil

print("📤 Upload diabetic_data.csv...")
uploaded = files.upload()  # Select diabetic_data.csv from your computer

# Move to correct location
data_filename = list(uploaded.keys())[0]
os.makedirs("data/raw", exist_ok=True)
shutil.move(data_filename, "data/raw/diabetic_data.csv")

print("✅ Data file moved to data/raw/diabetic_data.csv")

# Verify
!ls -lh data/raw/
```

### Option C: Data on Google Drive

If data is on Google Drive:

```python
# Mount Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Copy data file from Drive to project
import shutil
import os

os.makedirs("data/raw", exist_ok=True)
shutil.copy(
    "/content/drive/MyDrive/path/to/diabetic_data.csv",  # Adjust path
    "data/raw/diabetic_data.csv"
)

print("✅ Data copied from Google Drive")
```

---

## ▶️ Running the Pipeline in Colab

### Method 1: Direct Execution Script (Recommended)

```python
# Run the complete pipeline
!python run_pipeline_direct.py
```

**Expected output**: Full pipeline execution with progress bars, metrics, and visualizations

**Runtime**: ~3-5 minutes on Colab free tier (CPU)

### Method 2: Execute Individual Notebooks

**Option 2a**: Run notebooks programmatically

```python
# Execute notebooks in sequence
!python run_pipeline.py
```

**Option 2b**: Open and run notebooks interactively

1. In Colab, go to: `File` → `Upload notebook`
2. Navigate to `notebooks/` folder in your project
3. Upload `01_eda.ipynb`, `02_baseline_if.ipynb`, etc.
4. Run cells manually

**Note**: Each notebook has a path setup cell at the top:
```python
# Universal path setup (works in both Colab and local)
import sys
from pathlib import Path

# Detect environment
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# Add src to path
if IN_COLAB:
    project_root = Path('/content/ontology-anomaly-detection')  # Adjust if needed
    src_path = project_root / 'src'
else:
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    src_path = project_root / 'src'

sys.path.insert(0, str(src_path))

# Verify imports work
from preprocessing import load_raw_data, build_feature_matrix
from models import IsolationForestDetector, AutoencoderDetector
from ontology import compute_ontology_penalty, combine_scores
from evaluation import evaluate_anomaly_detector, plot_roc_pr_curves

print("✅ All imports successful!")
```

---

## 📁 Viewing Results in Colab

### List Generated Files

```python
# Check results directory
import os
from pathlib import Path

results_dir = Path("results")

if results_dir.exists():
    print("📊 Generated Results:\n")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(str(results_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            size_kb = file_path.stat().st_size / 1024
            print(f"{subindent}{file} ({size_kb:.1f} KB)")
else:
    print("❌ No results directory found")
```

### View Images in Colab

```python
from IPython.display import Image, display
import os

# Display all figures
figures_dir = "results/figures"
if os.path.exists(figures_dir):
    for filename in sorted(os.listdir(figures_dir)):
        if filename.endswith('.png'):
            print(f"\n📊 {filename}:")
            display(Image(filename=os.path.join(figures_dir, filename)))
else:
    print("❌ No figures directory found")
```

### Download Results

```python
# Download all results as a zip
!zip -r results.zip results/

from google.colab import files
files.download('results.zip')
```

---

## 🔧 Troubleshooting Colab

### Error: `ModuleNotFoundError: No module named 'src'`

**Cause**: Path not set up correctly

**Solution**:
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Verify
import sys
print("Python path:")
for p in sys.path:
    print(f"  - {p}")
```

### Error: `FileNotFoundError: 'data/raw/diabetic_data.csv'`

**Cause**: Data file not uploaded or in wrong location

**Solution**:
```python
# Check current directory
!pwd

# Check if data exists
!ls -la data/raw/

# If missing, upload data (see "Ensuring Data is Available" section above)
```

### Error: `No module named 'pandas'` (or any dependency)

**Cause**: Dependencies not installed

**Solution**:
```python
# Reinstall dependencies
!pip install --upgrade pip
!pip install -r requirements.txt
```

### Error: `RuntimeError: CUDA out of memory`

**Cause**: Autoencoder training on GPU, but Colab GPU has limited memory

**Solution**:
```python
# Option 1: Use smaller batch size (edit in src/models.py or notebook)
# Option 2: Force CPU usage
import torch
device = torch.device('cpu')  # Instead of 'cuda'
```

### Error: Notebooks fail to import from `src/`

**Cause**: Path setup cell not executed or incorrect

**Solution**:
- Make sure to run the path setup cell (first cell) in every notebook
- Verify you're in the project root:
  ```python
  import os
  print(f"Current directory: {os.getcwd()}")
  # Should end with your project folder name
  ```

---

## ⚡ Colab Pro Tips

### 1. Enable GPU (Optional - Faster Autoencoder Training)

- Click "Runtime" → "Change runtime type"
- Hardware accelerator: Select "GPU" (T4)
- Click "Save"
- **Note**: PyTorch autoencoder will automatically use GPU if available

### 2. Keep Session Alive

Colab disconnects after ~90 minutes of inactivity:

```python
# Run this in a cell to print progress and keep session alive
import time
for i in range(100):
    print(f"Heartbeat {i}: Pipeline running...")
    time.sleep(60)  # Every minute
```

### 3. Save Outputs to Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r results/ "/content/drive/MyDrive/anomaly_detection_results/"
print("✅ Results backed up to Google Drive")
```

### 4. Persistent Environment (Reconnect Later)

To avoid re-uploading:
- **Use Option 3** (Google Drive mount) - just mount and navigate each time
- **OR** upload once, then use "Copy to Drive" to save the entire folder

---

## ✅ Colab Execution Checklist

Before running the pipeline, verify:

- [ ] Project folder accessible (`!ls -la` shows `data/`, `src/`, `notebooks/`, `requirements.txt`)
- [ ] Dependencies installed (`!pip list | grep pandas` shows installed version)
- [ ] Data file exists (`!ls data/raw/diabetic_data.csv` shows file)
- [ ] Imports work (`from preprocessing import load_raw_data` runs without error)

If all checked → **Run `!python run_pipeline_direct.py`** 🚀

---

## 🆚 Colab vs Local: Which Should You Use?

| Feature | Google Colab | Local (Miniconda) |
|---------|--------------|-------------------|
| **Setup Time** | ~2 minutes | ~10 minutes (first time) |
| **Internet Required** | Yes (always) | Only for installation |
| **Free GPU** | ✅ Yes (T4) | ❌ No (unless you have one) |
| **Data Privacy** | ⚠️ Uploads to Google | ✅ Stays on your machine |
| **Persistent Environment** | ❌ Resets every session | ✅ Permanent until deleted |
| **File Storage** | ⚠️ Lost on disconnect (unless saved to Drive) | ✅ Local disk |
| **Best For** | Quick tests, GPU training, demos | Development, repeated runs, privacy |

**Recommendation**:
- **Use Colab**: First-time testing, presentation demos, or if you have environment issues locally
- **Use Local**: Serious development, repeated experiments, or offline work

---

## 📞 Need Help?

If Colab issues persist:
- Check runtime is not disconnected: Look for green checkmark next to "RAM" and "Disk" in top-right
- Restart runtime: "Runtime" → "Restart runtime"
- Try a different browser (Chrome works best with Colab)
- Check Colab status: https://status.cloud.google.com/

**Alternative**: Follow `LOCAL_WINDOWS_SETUP.md` for local installation
