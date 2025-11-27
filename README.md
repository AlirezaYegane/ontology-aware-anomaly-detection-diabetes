# Ontology-Aware Anomaly Detection for Diabetes Hospital Readmissions

A professional machine learning pipeline combining **anomaly detection** (Isolation Forest & Autoencoder) with **domain-knowledge ontology rules** to identify high-risk early hospital readmissions in diabetes patients.

## 📊 Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**
- **Source**: UCI Machine Learning Repository
- **Size**: 101,766 hospital encounters
- **Task**: Binary classification (readmitted <30 days vs not)
- **Features**: Demographics, hospital stay metrics, lab results, medication changes

## 🏗️ Project Structure

```
ontology-aware-anomaly-detection-diabetes/
├── data/
│   ├── raw/                   # diabetic_data.csv, IDS_mapping.csv
│   └── processed/             # Preprocessed outputs (generated)
├── notebooks/                 # 4 executable Jupyter notebooks
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_baseline_if.ipynb  # Isolation Forest baseline
│   ├── 03_autoencoder.ipynb  # PyTorch Autoencoder
│   └── 04_ontology_eval.ipynb # Ontology-enhanced evaluation
├── results/                   # Output plots and metrics (generated)
├── src/                       # Core Python modules
│   ├── __init__.py
│   ├── preprocessing.py       # Data loading, cleaning, feature engineering
│   ├── models.py              # Isolation Forest & Autoencoder models
│   ├── ontology.py            # Clinical rule-based penalty layer
│   └── evaluation.py          # Metrics, ROC/PR curves, visualization
├── archive/                   # Legacy code and documentation
├── TASKS.md                   # Project task checklist
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ontology-aware-anomaly-detection-diabetes
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place data files** (if not already present)
   - Download `diabetic_data.csv` and place it in `data/raw/`

4. **Run notebooks in order**
   ```bash
   jupyter notebook
   ```
   - Execute `01_eda.ipynb` → `02_baseline_if.ipynb` → `03_autoencoder.ipynb` → `04_ontology_eval.ipynb`

### Option 2: Run on Google Colab

1. **Upload the project folder** to your Google Drive

2. **Open any notebook** in Google Colab (e.g., `notebooks/01_eda.ipynb`)

3. **Mount Google Drive** and navigate to project directory:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/ontology-aware-anomaly-detection-diabetes
   ```

4. **Run the notebooks**  
   ✅ The first cell in each notebook contains a path fix for Colab compatibility:
   ```python
   import sys, os
   sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
   ```

## 📓 Notebook Workflow

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| **01_eda.ipynb** | Explore data distributions, correlations, missing values | Visualizations, data insights |
| **02_baseline_if.ipynb** | Train Isolation Forest, evaluate baseline performance | ROC/PR curves, metrics CSV |
| **03_autoencoder.ipynb** | Train PyTorch Autoencoder, compute reconstruction errors | Training loss plot, ROC/PR curves |
| **04_ontology_eval.ipynb** | Apply ontology rules, compare ML vs ML+Ontology | Comparison table, combined metrics |

## 🧬 Methodology

### 1. Anomaly Detection Models
- **Isolation Forest**: Unsupervised tree-based anomaly detector
- **Feedforward Autoencoder**: Neural network trained on normal samples (PyTorch)

### 2. Ontology-Inspired Rules
Clinical domain knowledge encoded as penalty scores:

| Rule | Condition | Penalty |
|------|-----------|---------|
| **High Risk** | Poor glycemic control (A1C>7/8) + No med change + On diabetes meds | 0.9 |
| **High Risk** | Very high glucose (>200/300) + Insufficient lab procedures (<40) | 0.85 |
| **Medium Risk** | High medication burden (>20) + Short stay (<3 days) | 0.6 |
| **Low Risk** | None of the above | 0.1 |

### 3. Score Combination
```
Final Score = α × Normalized(ML_Score) + β × Ontology_Penalty
```
Default: α=0.7 (ML weight), β=0.3 (Ontology weight)

## 📈 Expected Results

Results are saved to `results/` directory:
- `if_roc_pr_curves.png` - Isolation Forest performance
- `autoencoder_roc_pr_curves.png` - Autoencoder performance  
- `ontology_comparison_curves.png` - ML vs ML+Ontology comparison
- `*.csv` - Metrics summaries (ROC-AUC, PR-AUC)

**Performance Metrics**:
- **ROC-AUC**: Area under ROC curve (higher is better)
- **PR-AUC**: Precision-Recall AUC (important for imbalanced data)

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- pandas, numpy, matplotlib, seaborn
- Jupyter Notebook

See `requirements.txt` for full list.

## 📚 References

- **Dataset**: Beata Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, 2014.
- **Isolation Forest**: Liu et al., "Isolation Forest", ICDM 2008
- **Autoencoders for Anomaly Detection**: Sakurada & Yairi, "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction", MLSDA Workshop 2014

## 🎯 Future Work

- Explore additional autoencoder architectures (VAE, LSTM-AE)
- Tune ontology rule weights with expert feedback
- Add more clinical rules based on domain expertise
- Deploy as web service for real-time risk prediction
- Validate on external healthcare datasets

## 📄 License

This project is for educational and research purposes.

## 👤 Author

Created as a demonstration of combining machine learning with domain knowledge for healthcare anomaly detection.

---

**Note**: This project is GitHub-ready and optimized for Google Colab execution. All notebooks include path fixes for seamless imports from `src/` modules.
