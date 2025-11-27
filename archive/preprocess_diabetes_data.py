"""
Diabetes Hospital Readmission Data Preprocessing Script

Complete preprocessing pipeline for the UCI Diabetes 130-US Hospitals dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)
sns.set_style('whitegrid')

print("="*80)
print("DIABETES HOSPITAL READMISSION - DATA PREPROCESSING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv('data/diabetic_data.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 2. BASIC DATA INFORMATION
# ============================================================================
print("\n[2] Basic Data Information")
print("-" * 80)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"\nData Types:")
print(df.dtypes.value_counts())

# Check for missing values (including '?')
print("\n[3] Missing Values Analysis (counting '?' as missing)")
print("-" * 80)
missing_counts = (df == '?').sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)
if len(missing_df) > 0:
    print(missing_df.head(10))
else:
    print("No missing values found")

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n[4] Target Variable: 'readmitted' Analysis")
print("-" * 80)
print("Value Counts:")
print(df['readmitted'].value_counts())
print("\nPercentages:")
print(df['readmitted'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

# Visualize readmitted distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
readmit_counts = df['readmitted'].value_counts()
axes[0].bar(readmit_counts.index, readmit_counts.values, 
            color=['green', 'orange', 'red'], alpha=0.7)
axes[0].set_xlabel('Readmitted Category')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Readmission Status')
axes[0].grid(axis='y', alpha=0.3)
for i, (idx, val) in enumerate(readmit_counts.items()):
    axes[0].text(i, val, f'{val:,}', ha='center', va='bottom')

readmit_pct = df['readmitted'].value_counts(normalize=True) * 100
axes[1].bar(readmit_pct.index, readmit_pct.values, 
            color=['green', 'orange', 'red'], alpha=0.7)
axes[1].set_xlabel('Readmitted Category')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('Distribution of Readmission Status (%)')
axes[1].grid(axis='y', alpha=0.3)
for i, (idx, val) in enumerate(readmit_pct.items()):
    axes[1].text(i, val, f'{val:.1f}%', ha='center', va='bottom')

plt.tight_layout()
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'readmission_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_dir / 'readmission_distribution.png'}")
plt.close()

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================
print("\n[5] Feature Selection")
print("-" * 80)
selected_features = [
    # Demographics
    'race', 'gender', 'age',
    # Hospital stay metrics
    'time_in_hospital',
    # Procedure counts
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    # Outpatient/Emergency visits
    'number_outpatient',
    'number_inpatient',
    'number_emergency',
    # Lab results
    'A1Cresult',
    'max_glu_serum',
    # Medication changes
    'change',
    'diabetesMed'
]

missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    print(f"⚠ Warning: Missing features: {missing_features}")
else:
    print(f"✓ All {len(selected_features)} selected features are present")
    print(f"  Features: {', '.join(selected_features)}")

# Create subset
df_subset = df[selected_features + ['readmitted']].copy()
print(f"\nSubset shape: {df_subset.shape}")

# ============================================================================
# 5. DATA CLEANING
# ============================================================================
print("\n[6] Data Cleaning")
print("-" * 80)
print(f"Rows before cleaning: {len(df_subset):,}")

# Replace '?' with NaN
df_clean = df_subset.replace('?', np.nan)
print("✓ Replaced '?' with NaN")

# Show missing values
missing_info = df_clean.isnull().sum()
missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
if len(missing_info) > 0:
    print(f"\nMissing values per column:")
    print(missing_info)

# Drop rows with missing values
df_clean = df_clean.dropna()
print(f"\nRows after cleaning: {len(df_clean):,}")
print(f"Rows dropped: {len(df_subset) - len(df_clean):,} ({(len(df_subset) - len(df_clean))/len(df_subset)*100:.1f}%)")

# ============================================================================
# 6. CREATE BINARY TARGET
# ============================================================================
print("\n[7] Creating Binary Target Variable")
print("-" * 80)
# y = 1 if readmitted < 30 days, else 0
y = (df_clean['readmitted'] == '<30').astype(int)

print(f"y = 1 (readmitted < 30 days): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"y = 0 (not readmitted < 30):  {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"Class imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# Visualize binary target
plt.figure(figsize=(8, 5))
y_counts = y.value_counts()
plt.bar(['Not Readmitted <30', 'Readmitted <30'], 
        [y_counts[0], y_counts[1]], 
        color=['green', 'red'], alpha=0.7)
plt.ylabel('Count')
plt.title('Binary Classification Target Distribution')
plt.grid(axis='y', alpha=0.3)
for i, val in enumerate([y_counts[0], y_counts[1]]):
    plt.text(i, val, f'{val:,}\n({val/len(y)*100:.1f}%)', 
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig(output_dir / 'binary_target_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_dir / 'binary_target_distribution.png'}")
plt.close()

# ============================================================================
# 7. FEATURE ENCODING
# ============================================================================
print("\n[8] Feature Encoding and Preprocessing")
print("-" * 80)

# Separate features from target
X = df_clean.drop('readmitted', axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_cols)
    ],
    remainder='passthrough'
)

print("\n✓ Created preprocessing pipeline (StandardScaler + OneHotEncoder)")

# Fit and transform
X_transformed = preprocessor.fit_transform(X)

print(f"\nOriginal feature matrix shape: {X.shape}")
print(f"Transformed feature matrix shape: {X_transformed.shape}")
print(f"Features expanded (one-hot encoding): +{X_transformed.shape[1] - X.shape[1]}")

# Get feature names
feature_names = []
feature_names.extend(numerical_cols)
if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)

print(f"\nTotal features after encoding: {len(feature_names)}")

# Create final DataFrame
X_final = pd.DataFrame(
    X_transformed, 
    columns=feature_names,
    index=X.index
)

# ============================================================================
# 8. SAVE PREPROCESSED DATA
# ============================================================================
print("\n[9] Saving Preprocessed Data")
print("-" * 80)

processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

# Save features and target
X_final.to_csv(processed_dir / 'X_features.csv', index=False)
y.to_csv(processed_dir / 'y_target.csv', index=False, header=['target'])

# Save preprocessor
with open(processed_dir / 'preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"✓ Saved X_features.csv ({X_final.shape})")
print(f"✓ Saved y_target.csv ({y.shape})")
print(f"✓ Saved preprocessor.pkl")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL PREPROCESSED DATASET SUMMARY")
print("="*80)
print(f"\nSamples: {X_final.shape[0]:,}")
print(f"Features: {X_final.shape[1]:,}")
print(f"\nTarget distribution:")
print(f"  Class 0 (not readmitted <30): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  Class 1 (readmitted <30):     {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"\nFeature types:")
print(f"  Original numerical: {len(numerical_cols)}")
print(f"  Original categorical: {len(categorical_cols)}")
print(f"  After one-hot encoding: {X_final.shape[1]}")

print("\n" + "="*80)
print("✓ PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  • data/processed/X_features.csv")
print(f"  • data/processed/y_target.csv")
print(f"  • data/processed/preprocessor.pkl")
print(f"  • results/figures/readmission_distribution.png")
print(f"  • results/figures/binary_target_distribution.png")

# Display sample of processed data
print("\n" + "="*80)
print("SAMPLE OF PROCESSED DATA (first 5 rows, first 10 features):")
print("="*80)
print(X_final.iloc[:5, :10])

print("\n" + "="*80)
print("READY FOR MODELING!")
print("="*80)
print("\nNext steps:")
print("  1. Train/test split")
print("  2. Model training (Logistic Regression, Random Forest, XGBoost, etc.)")
print("  3. Model evaluation (ROC-AUC, Precision-Recall)")
print("  4. Feature importance analysis")
