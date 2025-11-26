"""
Complete End-to-End Anomaly Detection Pipeline
Combines preprocessing + anomaly detection + evaluation
"""

import sys
import subprocess
import importlib.util

# ============================================================================
# CHECK AND INSTALL DEPENDENCIES
# ============================================================================
def check_install_package(package_name, import_name=None):
    """Check if package exists, install if needed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"✓ {package_name} installed")
        return True
    return False

print("="*80)
print("CHECKING DEPENDENCIES")
print("="*80)

required_packages = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]

for pkg_name, import_name in required_packages:
    check_install_package(pkg_name, import_name)

print("\n✓ All dependencies ready!")

# ============================================================================
# NOW IMPORT EVERYTHING
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')

print("\n" + "="*80)
print("DIABETES READMISSION - COMPLETE PIPELINE")
print("="*80)

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("PART 1: DATA PREPROCESSING")
print("="*80)

# Load data
print("\n[1/7] Loading dataset...")
try:
    df = pd.read_csv('data/diabetic_data.csv')
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("ERROR: data/diabetic_data.csv not found!")
    print("Please ensure the dataset exists in the data/ directory")
    sys.exit(1)

# Feature selection
print("\n[2/7] Selecting features...")
selected_features = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_inpatient', 'number_emergency',
    'A1Cresult', 'max_glu_serum', 'change', 'diabetesMed'
]

df_subset = df[selected_features + ['readmitted']].copy()
print(f"✓ Selected {len(selected_features)} features")

# Clean data
print("\n[3/7] Cleaning data...")
print(f"  Before: {len(df_subset):,} rows")
df_clean = df_subset.replace('?', np.nan).dropna()
print(f"  After:  {len(df_clean):,} rows")
print(f"  Dropped: {len(df_subset) - len(df_clean):,} rows ({(len(df_subset) - len(df_clean))/len(df_subset)*100:.1f}%)")

# Create binary target
print("\n[4/7] Creating binary target...")
y = (df_clean['readmitted'] == '<30').astype(int)
print(f"  y=1 (readmitted <30): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  y=0 (not readmitted):  {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")

# Encode features
print("\n[5/7] Encoding features...")
X = df_clean.drop('readmitted', axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_cols)
    ]
)

X_transformed = preprocessor.fit_transform(X)
feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
X_final = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

print(f"  Original: {X.shape[1]} features")
print(f"  Encoded:  {X_final.shape[1]} features")

# Save preprocessed data
print("\n[6/7] Saving preprocessed data...")
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

X_final.to_csv(processed_dir / 'X_features.csv', index=False)
y.to_csv(processed_dir / 'y_target.csv', index=False, header=['target'])
with open(processed_dir / 'preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"✓ Saved to data/processed/")

# ============================================================================
# PART 2: ANOMALY DETECTION
# ============================================================================
print("\n" + "="*80)
print("PART 2: ANOMALY DETECTION WITH ISOLATION FOREST")
print("="*80)

# Train/test split
print("\n[1/6] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} samples")
print(f"  Test:  {len(X_test):,} samples")

# Train IsolationForest on normal samples only
print("\n[2/6] Training IsolationForest...")
X_train_normal = X_train[y_train == 0]
print(f"  Training on {len(X_train_normal):,} normal samples...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.11,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal)
print("✓ Model trained")

# Compute anomaly scores
print("\n[3/6] Computing anomaly scores...")
test_scores = -iso_forest.decision_function(X_test)
print(f"  Score range: [{test_scores.min():.3f}, {test_scores.max():.3f}]")

# Calculate metrics
print("\n[4/6] Evaluating performance...")
roc_auc = roc_auc_score(y_test, test_scores)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_scores)
pr_auc = auc(recall_vals, precision_vals)

print(f"  ROC-AUC:  {roc_auc:.4f}")
print(f"  PR-AUC:   {pr_auc:.4f}")

# Evaluate at different thresholds
print("\n  Performance at different thresholds:")
print("  " + "-"*60)
thresholds = [
    np.percentile(test_scores, 90),
    np.percentile(test_scores, 85),
    np.percentile(test_scores, 80)
]

for i, threshold in enumerate(thresholds):
    y_pred = (test_scores >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    flagged = (y_pred == 1).sum() / len(y_pred) * 100
    print(f"  Top {100-[90,85,80][i]}%: Precision={precision:.3f}, Recall={recall:.3f}, Flagged={flagged:.1f}%")

# Create visualizations
print("\n[5/6] Generating visualizations...")
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ROC and PR curves
fpr, tpr, _ = roc_curve(y_test, test_scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(fpr, tpr, linewidth=2, label=f'IsolationForest (AUC = {roc_auc:.3f})', color='#2E86AB')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
axes[0].set_title('ROC Curve - Anomaly Detection', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(alpha=0.3)

# PR Curve
axes[1].plot(recall_vals, precision_vals, linewidth=2, 
             label=f'IsolationForest (AUC = {pr_auc:.3f})', color='#A23B72')
baseline = (y_test == 1).sum() / len(y_test)
axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Baseline = {baseline:.3f}')
axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'isolation_forest_evaluation.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'isolation_forest_evaluation.png'}")

# Score distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(test_scores[y_test == 0], bins=50, alpha=0.6, label='Normal (y=0)', color='#06A77D')
axes[0].hist(test_scores[y_test == 1], bins=50, alpha=0.6, label='Anomaly (y=1)', color='#D90429')
axes[0].set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Anomaly Score Distribution by Class', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Box plot
bp = axes[1].boxplot([test_scores[y_test == 0], test_scores[y_test == 1]], 
                labels=['Normal (y=0)', 'Anomaly (y=1)'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], ['#06A77D', '#D90429']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
axes[1].set_title('Anomaly Score Box Plot', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'anomaly_score_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'anomaly_score_distributions.png'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\n📊 RESULTS SUMMARY")
print("-"*80)
print(f"Dataset:          {len(X_final):,} samples, {X_final.shape[1]} features")
print(f"Train/Test Split: {len(X_train):,} / {len(X_test):,}")
print(f"Anomaly Rate:     {(y==1).sum()/len(y)*100:.1f}%")
print(f"\nModel Performance:")
print(f"  ROC-AUC:        {roc_auc:.4f}")
print(f"  PR-AUC:         {pr_auc:.4f}")

print("\n📁 OUTPUT FILES")
print("-"*80)
print("Preprocessed Data:")
print("  ✓ data/processed/X_features.csv")
print("  ✓ data/processed/y_target.csv")
print("  ✓ data/processed/preprocessor.pkl")
print("\nVisualizations:")
print("  ✓ results/figures/isolation_forest_evaluation.png")
print("  ✓ results/figures/anomaly_score_distributions.png")

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*80)
