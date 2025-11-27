"""
Example Usage: Ontology-Inspired Anomaly Detection
====================================================

This script demonstrates how to use the ontology_rule_layer module
to enhance anomaly detection with domain-knowledge-based rules.

Author: ML Research Team
Dataset: Diabetes 130-US Hospitals (1999-2008)
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
]

for pkg_name, import_name in required_packages:
    check_install_package(pkg_name, import_name)

print("\n✓ All dependencies ready!")

# ============================================================================
# IMPORT MODULES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our ontology rule layer
from ontology_rule_layer import compare_anomaly_methods, compute_ontology_penalty

pd.set_option('display.max_columns', 50)

print("\n" + "="*80)
print("ONTOLOGY-INSPIRED ANOMALY DETECTION - COMPLETE EXAMPLE")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

# Load dataset
print("\n[1/5] Loading Diabetes 130-US dataset...")
try:
    df = pd.read_csv('data/diabetic_data.csv')
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("ERROR: data/diabetic_data.csv not found!")
    print("Please ensure the dataset exists in the data/ directory")
    sys.exit(1)

# Select relevant features
print("\n[2/5] Selecting features for analysis...")
selected_features = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_inpatient', 'number_emergency',
    'A1Cresult', 'max_glu_serum', 'change', 'diabetesMed'
]

df_subset = df[selected_features + ['readmitted']].copy()
print(f"✓ Selected {len(selected_features)} features")

# Clean data
print("\n[3/5] Cleaning data...")
print(f"  Before: {len(df_subset):,} rows")
df_clean = df_subset.replace('?', np.nan).dropna()
print(f"  After:  {len(df_clean):,} rows")
print(f"  Dropped: {len(df_subset) - len(df_clean):,} rows ({(len(df_subset) - len(df_clean))/len(df_subset)*100:.1f}%)")

# Create binary target (1 = readmitted within 30 days, 0 = otherwise)
print("\n[4/5] Creating binary target variable...")
y = (df_clean['readmitted'] == '<30').astype(int)
print(f"  y=1 (early readmission): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  y=0 (no early readmit):  {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")

# Encode and scale features
print("\n[5/5] Encoding and scaling features...")
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

# ============================================================================
# STEP 2: TRAIN ISOLATION FOREST BASELINE
# ============================================================================
print("\n" + "="*80)
print("STEP 2: BASELINE ANOMALY DETECTION (ISOLATION FOREST)")
print("="*80)

# Train/test split
print("\n[1/3] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} samples")
print(f"  Test:  {len(X_test):,} samples")

# Train IsolationForest on normal samples only
print("\n[2/3] Training IsolationForest on normal samples...")
X_train_normal = X_train[y_train == 0]
print(f"  Using {len(X_train_normal):,} normal samples for training...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.11,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal)
print("✓ IsolationForest model trained")

# Compute anomaly scores on test set
print("\n[3/3] Computing anomaly scores on test set...")
test_scores_raw = -iso_forest.decision_function(X_test)

# Normalize scores to [0, 1] range for easier interpretation
min_score = test_scores_raw.min()
max_score = test_scores_raw.max()
test_scores_normalized = (test_scores_raw - min_score) / (max_score - min_score)

print(f"  Raw score range: [{test_scores_raw.min():.3f}, {test_scores_raw.max():.3f}]")
print(f"  Normalized range: [{test_scores_normalized.min():.3f}, {test_scores_normalized.max():.3f}]")

# ============================================================================
# STEP 3: PREPARE TEST DATAFRAME WITH ORIGINAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 3: PREPARING TEST DATA WITH ORIGINAL FEATURES")
print("="*80)

print("\nCombining test data with original features...")
# Get the original features for test samples
df_test_original = df_clean.loc[X_test.index].copy()

# Add IsolationForest scores and target
df_test_original['iforest_score'] = test_scores_normalized
df_test_original['y'] = y_test.values

print(f"✓ Test dataframe prepared with {len(df_test_original)} samples")
print(f"  Columns: {list(df_test_original.columns)}")

# ============================================================================
# STEP 4: APPLY ONTOLOGY-INSPIRED RULE LAYER
# ============================================================================
print("\n" + "="*80)
print("STEP 4: APPLYING ONTOLOGY-INSPIRED RULE LAYER")
print("="*80)
print()

# This is the main function call that applies ontology rules and evaluates!
df_test_enhanced = compare_anomaly_methods(df_test_original)

# ============================================================================
# STEP 5: INSPECT RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: DETAILED INSPECTION OF RESULTS")
print("="*80)

print("\n[1] Sample of enhanced test data with all scores:")
print("-"*80)
sample_cols = ['iforest_score', 'ontology_penalty', 'final_score', 'y', 
               'A1Cresult', 'max_glu_serum', 'num_medications', 'time_in_hospital']
print(df_test_enhanced[sample_cols].head(10))

print("\n[2] Records with high ontology penalty (>0.8):")
print("-"*80)
high_penalty = df_test_enhanced[df_test_enhanced['ontology_penalty'] > 0.8]
print(f"Found {len(high_penalty)} records with high ontology penalty")
if len(high_penalty) > 0:
    print("\nSample high-penalty records:")
    print(high_penalty[sample_cols].head())
    
print("\n[3] Records flagged by ontology but not by IsolationForest:")
print("-"*80)
ontology_only = df_test_enhanced[
    (df_test_enhanced['ontology_penalty'] > 0.6) & 
    (df_test_enhanced['iforest_score'] < df_test_enhanced['iforest_score'].median())
]
print(f"Found {len(ontology_only)} records flagged by ontology rules but not IForest")
if len(ontology_only) > 0:
    print("\nSample ontology-only anomalies:")
    print(ontology_only[sample_cols].head())

print("\n[4] Records with discrepancy between ML and ontology:")
print("-"*80)
# High IForest but low ontology
ml_only = df_test_enhanced[
    (df_test_enhanced['iforest_score'] > df_test_enhanced['iforest_score'].quantile(0.9)) &
    (df_test_enhanced['ontology_penalty'] < 0.3)
]
print(f"High ML score, low ontology: {len(ml_only)} records")

# Low IForest but high ontology
ontology_high_ml_low = df_test_enhanced[
    (df_test_enhanced['iforest_score'] < df_test_enhanced['iforest_score'].quantile(0.5)) &
    (df_test_enhanced['ontology_penalty'] > 0.7)
]
print(f"Low ML score, high ontology: {len(ontology_high_ml_low)} records")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*80)

output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Visualization 1: Score distributions comparison
print("\n[1/3] Creating score distribution comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# IForest score by class
axes[0, 0].hist(df_test_enhanced[df_test_enhanced['y']==0]['iforest_score'], 
                bins=40, alpha=0.6, label='Normal (y=0)', color='#06A77D')
axes[0, 0].hist(df_test_enhanced[df_test_enhanced['y']==1]['iforest_score'], 
                bins=40, alpha=0.6, label='Readmit (y=1)', color='#D90429')
axes[0, 0].set_xlabel('IsolationForest Score', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('IsolationForest Score Distribution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Ontology penalty by class
axes[0, 1].hist(df_test_enhanced[df_test_enhanced['y']==0]['ontology_penalty'], 
                bins=40, alpha=0.6, label='Normal (y=0)', color='#06A77D')
axes[0, 1].hist(df_test_enhanced[df_test_enhanced['y']==1]['ontology_penalty'], 
                bins=40, alpha=0.6, label='Readmit (y=1)', color='#D90429')
axes[0, 1].set_xlabel('Ontology Penalty Score', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Ontology Penalty Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Final combined score by class
axes[1, 0].hist(df_test_enhanced[df_test_enhanced['y']==0]['final_score'], 
                bins=40, alpha=0.6, label='Normal (y=0)', color='#06A77D')
axes[1, 0].hist(df_test_enhanced[df_test_enhanced['y']==1]['final_score'], 
                bins=40, alpha=0.6, label='Readmit (y=1)', color='#D90429')
axes[1, 0].set_xlabel('Final Combined Score', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Final Combined Score Distribution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Scatter: IForest vs Ontology, colored by true label
scatter_colors = ['#06A77D' if y == 0 else '#D90429' for y in df_test_enhanced['y']]
axes[1, 1].scatter(df_test_enhanced['iforest_score'], 
                  df_test_enhanced['ontology_penalty'],
                  c=scatter_colors, alpha=0.5, s=20)
axes[1, 1].set_xlabel('IsolationForest Score', fontweight='bold')
axes[1, 1].set_ylabel('Ontology Penalty', fontweight='bold')
axes[1, 1].set_title('IForest vs Ontology Penalty', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#06A77D', label='Normal (y=0)'),
                  Patch(facecolor='#D90429', label='Readmit (y=1)')]
axes[1, 1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(output_dir / 'ontology_score_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'ontology_score_distributions.png'}")

# Visualization 2: Box plots
print("\n[2/3] Creating box plot comparisons...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (col, title) in enumerate([
    ('iforest_score', 'IsolationForest Score'),
    ('ontology_penalty', 'Ontology Penalty'),
    ('final_score', 'Final Combined Score')
]):
    bp = axes[idx].boxplot(
        [df_test_enhanced[df_test_enhanced['y']==0][col],
         df_test_enhanced[df_test_enhanced['y']==1][col]],
        labels=['Normal (y=0)', 'Readmit (y=1)'],
        patch_artist=True
    )
    for patch, color in zip(bp['boxes'], ['#06A77D', '#D90429']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[idx].set_ylabel('Score', fontweight='bold')
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'ontology_boxplot_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'ontology_boxplot_comparison.png'}")

# Visualization 3: Correlation heatmap
print("\n[3/3] Creating correlation heatmap...")
import seaborn as sns

score_cols = ['iforest_score', 'ontology_penalty', 'final_score', 'y']
corr_matrix = df_test_enhanced[score_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Score Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_dir / 'ontology_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir / 'ontology_correlation_heatmap.png'}")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

reports_dir = Path('results/reports')
reports_dir.mkdir(parents=True, exist_ok=True)

# Save enhanced test data
output_file = reports_dir / 'ontology_enhanced_predictions.csv'
df_test_enhanced.to_csv(output_file, index=False)
print(f"\n✓ Saved enhanced predictions to: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE COMPLETE!")
print("="*80)

print("\n📊 SUMMARY")
print("-"*80)
print(f"Total test samples:     {len(df_test_enhanced):,}")
print(f"True readmissions:      {(df_test_enhanced['y']==1).sum():,} ({(df_test_enhanced['y']==1).sum()/len(df_test_enhanced)*100:.1f}%)")
print(f"\nScore Statistics:")
print(f"  IForest mean:         {df_test_enhanced['iforest_score'].mean():.3f}")
print(f"  Ontology mean:        {df_test_enhanced['ontology_penalty'].mean():.3f}")
print(f"  Combined mean:        {df_test_enhanced['final_score'].mean():.3f}")

print("\n📁 OUTPUT FILES")
print("-"*80)
print("Data:")
print(f"  ✓ {reports_dir / 'ontology_enhanced_predictions.csv'}")
print("\nVisualizations:")
print(f"  ✓ {output_dir / 'ontology_score_distributions.png'}")
print(f"  ✓ {output_dir / 'ontology_boxplot_comparison.png'}")
print(f"  ✓ {output_dir / 'ontology_correlation_heatmap.png'}")

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("  1. Review the evaluation metrics in the console output above")
print("  2. Examine the visualizations in results/figures/")
print("  3. Analyze the enhanced predictions in results/reports/")
print("  4. Consider refining the ontology rules based on domain expertise")
print("\n" + "="*80)
