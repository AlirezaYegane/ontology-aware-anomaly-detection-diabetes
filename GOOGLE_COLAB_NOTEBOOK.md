# Google Colab Notebook - Diabetes Anomaly Detection

## 🚀 No Installation Required!

Copy this entire code into a Google Colab notebook and run it.

## Instructions

1. Go to https://colab.research.google.com/
2. Create new notebook
3. Upload `diabetic_data.csv` to Colab using the files panel
4. Paste and run the code below

---

## Complete Code for Google Colab

```python
# ============================================================================
# CELL 1: Install and Import
# ============================================================================
!pip install -q pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             auc, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
print("✓ All libraries imported!")

# ============================================================================
# CELL 2: Load Data
# ============================================================================
# Upload diabetic_data.csv using the files panel on the left first!
df = pd.read_csv('diabetic_data.csv')
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()

# ============================================================================
# CELL 3: Preprocessing
# ============================================================================
print("="*60)
print("DATA PREPROCESSING")
print("="*60)

# Select features
selected_features = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_inpatient', 'number_emergency',
    'A1Cresult', 'max_glu_serum', 'change', 'diabetesMed'
]

df_subset = df[selected_features + ['readmitted']].copy()
print(f"\\nSelected {len(selected_features)} features")

# Clean data
df_clean = df_subset.replace('?', np.nan).dropna()
print(f"Cleaned: {len(df_subset):,} → {len(df_clean):,} rows")

# Create binary target
y = (df_clean['readmitted'] == '<30').astype(int)
print(f"\\nTarget distribution:")
print(f"  y=1 (readmitted <30): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  y=0 (not readmitted):  {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")

# Encode features
X = df_clean.drop('readmitted', axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_transformed = preprocessor.fit_transform(X)
feature_names = numerical_cols + list(
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
)
X_final = pd.DataFrame(X_transformed, columns=feature_names)

print(f"\\nFinal: {X_final.shape[0]:,} samples × {X_final.shape[1]} features")

# ============================================================================
# CELL 4: Train/Test Split
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# CELL 5: Train Isolation Forest
# ============================================================================
print("="*60)
print("ANOMALY DETECTION")
print("="*60)

# Train on normal samples only
X_train_normal = X_train[y_train == 0]
print(f"\\nTraining on {len(X_train_normal):,} normal samples...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.11,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal)
print("✓ Model trained!")

# Compute scores
test_scores = -iso_forest.decision_function(X_test)
print(f"Anomaly scores computed")

# ============================================================================
# CELL 6: Evaluation
# ============================================================================
# Calculate metrics
roc_auc = roc_auc_score(y_test, test_scores)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_scores)
pr_auc = auc(recall_vals, precision_vals)

print(f"\\nPerformance:")
print(f"  ROC-AUC:  {roc_auc:.4f}")
print(f"  PR-AUC:   {pr_auc:.4f}")

# Threshold analysis
print(f"\\nThreshold Analysis:")
for percentile in [90, 85, 80]:
    threshold = np.percentile(test_scores, percentile)
    y_pred = (test_scores >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    flagged = (y_pred == 1).sum() / len(y_pred) * 100
    print(f"  Top {100-percentile}%: Precision={precision:.3f}, Recall={recall:.3f}, Flagged={flagged:.1f}%")

# ============================================================================
# CELL 7: Visualizations
# ============================================================================
# ROC and PR curves
fpr, tpr, _ = roc_curve(y_test, test_scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate', fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontweight='bold')
axes[0].set_title('ROC Curve', fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(alpha=0.3)

# PR
axes[1].plot(recall_vals, precision_vals, linewidth=2, label=f'AUC = {pr_auc:.3f}')
baseline = (y_test == 1).sum() / len(y_test)
axes[1].axhline(baseline, linestyle='--', color='k', alpha=0.5, label=f'Baseline = {baseline:.3f}')
axes[1].set_xlabel('Recall', fontweight='bold')
axes[1].set_ylabel('Precision', fontweight='bold')
axes[1].set_title('Precision-Recall Curve', fontweight='bold', fontsize=13)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_curves.png', dpi=200, bbox_inches='tight')
plt.show()

# Score distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(test_scores[y_test == 0], bins=50, alpha=0.6, label='Normal', color='green')
axes[0].hist(test_scores[y_test == 1], bins=50, alpha=0.6, label='Anomaly', color='red')
axes[0].set_xlabel('Anomaly Score', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Score Distribution by Class', fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(alpha=0.3)

bp = axes[1].boxplot([test_scores[y_test == 0], test_scores[y_test == 1]], 
                     labels=['Normal', 'Anomaly'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'red']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Anomaly Score', fontweight='bold')
axes[1].set_title('Score Box Plot', fontweight='bold', fontsize=13)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('score_distributions.png', dpi=200, bbox_inches='tight')
plt.show()

print("\\n✅ COMPLETE! Download the PNG files from the files panel.")
```

---

## Expected Output

After running all cells, you'll get:

### Console Output
```
✓ All libraries imported!
Loaded 101,766 rows × 50 columns

DATA PREPROCESSING
Selected 14 features
Cleaned: 101,766 → ~98,000 rows

Target distribution:
  y=1 (readmitted <30): ~11,000 (11.0%)
  y=0 (not readmitted):  ~87,000 (89.0%)

Final: ~98,000 samples × 35-40 features
Train: ~78,000 | Test: ~19,000

ANOMALY DETECTION
Training on ~70,000 normal samples...
✓ Model trained!
Anomaly scores computed

Performance:
  ROC-AUC:  0.6XXX
  PR-AUC:   0.1XXX

Threshold Analysis:
  Top 10%: Precision=0.XXX, Recall=0.XXX, Flagged=10.0%
  Top 15%: Precision=0.XXX, Recall=0.XXX, Flagged=15.0%
  Top 20%: Precision=0.XXX, Recall=0.XXX, Flagged=20.0%

✅ COMPLETE! Download the PNG files from the files panel.
```

### Generated Files
- `evaluation_curves.png` - ROC and PR curves
- `score_distributions.png` - Anomaly score distributions

---

## ⏱️ Execution Time

Total runtime: ~2-3 minutes on Google Colab free tier

## 💾 Download Results

1. Click folder icon on left sidebar
2. Right-click each PNG file
3. Select "Download"

---

**This is the easiest way to run the complete pipeline with zero installation!** 🚀
