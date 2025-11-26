"""
Preprocessing module for Diabetes Hospital Readmission dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path
from typing import Tuple, List, Optional

def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

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

def clean_data(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Clean the dataset:
    1. Select specific features + target ('readmitted')
    2. Replace '?' with NaN
    3. Drop rows with missing values
    """
    # Create subset
    df_subset = df[selected_features + ['readmitted']].copy()
    
    # Replace '?' with NaN
    df_clean = df_subset.replace('?', np.nan)
    
    # Drop rows with missing values
    df_clean = df_clean.dropna()
    
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
