"""
Preprocessing module for Diabetes Hospital Readmission dataset.

Includes comprehensive filtering instrumentation to track data transformations
from raw dataset to final feature matrix.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from typing import Tuple, List, Optional
from .filter_tracker import FilterTracker

def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.
    Alias for load_data() - specification-compliant name.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    return load_data(filepath)

def get_selected_features() -> List[str]:
    """Return the list of features selected for the model."""
    return [
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

def clean_data(df: pd.DataFrame, selected_features: List[str], 
               tracker: Optional[FilterTracker] = None) -> pd.DataFrame:
    """
    Clean the dataset:
    1. Select specific features + target ('readmitted')
    2. Replace '?' with NaN
    3. Drop rows with missing values in CRITICAL columns only

    FILTERING RATIONALE:
    - Feature selection: Reduces from 50 columns to 14 relevant clinical features.
      This is domain-driven - we select features that are clinically meaningful
      for predicting readmission.

    - Missing value handling: The dataset uses '?' to denote missing values.
      We convert these to NaN for proper handling.

    - Row removal:
      We ONLY drop rows that have missing values in a set of CRITICAL columns
      (target + key clinical features). Non-critical columns are kept and
      will be imputed later (numerical: median, categorical: 'Unknown').

      This avoids the previous behaviour where dropna() removed ~99.7% of
      the dataset.

    Args:
        df: Input DataFrame
        selected_features: List of feature names to keep
        tracker: Optional FilterTracker to instrument filtering steps

    Returns:
        Cleaned DataFrame (may still contain NaNs in non-critical columns)
    """
    df_start = df.copy()

    # Step 1: Select subset of columns (feature selection)
    df_subset = df[selected_features + ['readmitted']].copy()

    if tracker:
        tracker.track_step(
            df_start,
            df_subset,
            step_name="feature_selection",
            description=f"Select {len(selected_features)} features (from {len(df.columns)} columns)"
        )

    # Step 2: Replace '?' with NaN
    df_before_replace = df_subset.copy()
    df_clean = df_subset.replace('?', np.nan)

    if tracker:
        # Count how many cells were affected
        n_question_marks = (df_before_replace == '?').sum().sum()
        tracker.track_step(
            df_before_replace, df_clean,
            step_name="replace_missing_markers",
            description=f"Replace '?' with NaN ({n_question_marks:,} cells affected)"
        )

    # Step 3: Drop rows with missing values in CRITICAL columns only
    # Critical columns = target + key clinical features that must not be missing
    critical_cols = [
        'readmitted',
        'race',
        'gender',
        'age',
        'time_in_hospital',
        'num_lab_procedures',
        'num_medications',
        'number_inpatient',
        'number_emergency',
        'change',
        'diabetesMed',
    ]
    # Keep only those that actually exist in the current DataFrame
    critical_cols = [c for c in critical_cols if c in df_clean.columns]

    df_before_drop = df_clean.copy()
    df_clean = df_clean.dropna(subset=critical_cols)

    if tracker:
        n_removed = len(df_before_drop) - len(df_clean)
        tracker.track_step(
            df_before_drop, df_clean,
            step_name="drop_missing_values",
            description=(
                f"Drop rows with missing values in critical columns "
                f"{critical_cols} ({n_removed:,} rows removed)"
            )
        )

    return df_clean


def create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create binary target variable:
    y = 1 if readmitted < 30 days, else 0
    Returns X (features) and y (target)
    """
    y = (df['readmitted'] == '<30').astype(int)
    X = df.drop('readmitted', axis=1)
    return X, y

def get_feature_names(preprocessor: ColumnTransformer, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Extract feature names from the column transformer."""
    feature_names = []
    feature_names.extend(numerical_cols)
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    return feature_names

def fit_transform_data(X: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Apply OneHotEncoding and StandardScaler to the features.
    Returns the transformed DataFrame and the fitted preprocessor.
    """
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names and create DataFrame
    feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols)
    
    X_final = pd.DataFrame(
        X_transformed, 
        columns=feature_names,
        index=X.index
    )
    
    return X_final, preprocessor

def build_feature_matrix(
    data_or_path,
    save_preprocessor: bool = False,
    output_dir: Optional[Path] = None,
    enable_tracking: bool = True,
    save_summary: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[ColumnTransformer]]:
    """
    Complete preprocessing pipeline: load, clean, transform data.
    
    NOW WITH COMPREHENSIVE FILTERING INSTRUMENTATION!
    Tracks every step that modifies the dataset and generates detailed reports.

    Flexible API:
      - If you pass a pandas DataFrame  -> use it directly
      - If you pass a string / Path    -> treat it as CSV filepath and load it
    
    Args:
        data_or_path: DataFrame or path to CSV file
        save_preprocessor: Whether to save the fitted preprocessor
        output_dir: Directory for saving preprocessor
        enable_tracking: Enable FilterTracker instrumentation (default: True)
        save_summary: Save filtering summary to results/reports/ (default: True)
    
    Returns:
        X_transformed: Transformed feature matrix
        y: Binary target variable (1 = readmitted <30 days)
        preprocessor: Fitted ColumnTransformer (or None)
    """
    
    # Initialize FilterTracker if enabled
    tracker = FilterTracker() if enable_tracking else None
    
    if tracker:
        print("\n" + "="*80)
        print("  DATA FILTERING & PREPROCESSING PIPELINE")
        print("="*80 + "\n")

    # Step 1: Load or accept data
    if isinstance(data_or_path, pd.DataFrame):
        # Already a DataFrame: use it directly
        df = data_or_path.copy()
    else:
        # Assume this is a filepath (str or Path)
        df = load_raw_data(str(data_or_path))
    
    # Track initial dataset
    df_initial = df.copy()
    if tracker:
        tracker.track_step(
            df_initial, df_initial,
            step_name="load_raw_data",
            description=f"Load raw dataset ({df.shape[0]:,} rows × {df.shape[1]} columns)"
        )

    # Step 2: Get selected features
    selected_features = get_selected_features()

    # Step 3: Clean data (with tracking)
    df_clean = clean_data(df, selected_features, tracker=tracker)

    # Step 4: Create target variable (X features, y binary target)
    X_before_target = df_clean.drop('readmitted', axis=1)
    X, y = create_target(df_clean)
    
    if tracker:
        # Track target creation (typically no row removal, but good to confirm)
        tracker.track_step(
            X_before_target, X,
            step_name="create_target",
            description=f"Create binary target (readmitted <30 days)"
        )

     # Step 5: Impute remaining missing values in features (non-critical columns)
    X_imputed = impute_missing_values(X)

    if tracker:
        tracker.track_step(
            X, X_imputed,
            step_name="impute_missing_values",
            description="Impute missing values (numerical: median, categorical: 'Unknown')"
        )

    # Step 6: Fit and transform features
    X_transformed, preprocessor = fit_transform_data(X_imputed)
    
    if tracker:
        tracker.track_step(
            X_imputed, X_transformed,
            step_name="encode_and_scale",
            description=f"OneHotEncode + StandardScale ({X_imputed.shape[1]} -> {X_transformed.shape[1]} features)"
        )
        
        # Track final class balance
        tracker.track_class_balance(y)

    # Step 6: Optionally save preprocessor
    if save_preprocessor and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    
    # Step 7: Generate and save summary reports
    if tracker:
        # Print to console
        tracker.print_summary_table()
        
        # Save to files if requested
        if save_summary:
            # Determine output directory
            if output_dir:
                reports_dir = Path(output_dir) / 'reports'
            else:
                # Default to results/reports in current working directory
                reports_dir = Path('results') / 'reports'
            
            tracker.save_summary(reports_dir, basename="data_filtering_summary")

    return X_transformed, y, preprocessor

def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, transparent imputation for remaining missing values:

    - Numerical columns: fill NaN with column median
    - Categorical columns: fill NaN with the string 'Unknown'

    This is intentionally simple and explicit so that the behaviour is easy to
    reason about and describe in documentation.
    """
    X_imputed = X.copy()

    # Numerical columns -> median
    num_cols = X_imputed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X_imputed[col].isna().any():
            median = X_imputed[col].median()
            X_imputed[col] = X_imputed[col].fillna(median)

    # Categorical columns -> 'Unknown'
    cat_cols = X_imputed.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X_imputed[col].isna().any():
            X_imputed[col] = X_imputed[col].fillna('Unknown')

    return X_imputed


def train_test_split_stratified(X: pd.DataFrame, y: pd.Series, 
                                test_size: float = 0.2, 
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split.
    Specification-compliant function for splitting data.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def save_processed_data(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, output_dir: Path):
    """Save the processed data and preprocessor to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X.to_csv(output_dir / 'X_features.csv', index=False)
    y.to_csv(output_dir / 'y_target.csv', index=False, header=['target'])
    
    with open(output_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

if __name__ == "__main__":
    # Example usage
    print("Running preprocessing...")
    data_path = 'data/diabetic_data.csv'
    if Path(data_path).exists():
        df = load_data(data_path)
        features = get_selected_features()
        df_clean = clean_data(df, features)
        X, y = create_target(df_clean)
        X_processed, preprocessor = fit_transform_data(X)
        save_processed_data(X_processed, y, preprocessor, Path('data/processed'))
        print("Done!")
    else:
        print(f"File not found: {data_path}")
