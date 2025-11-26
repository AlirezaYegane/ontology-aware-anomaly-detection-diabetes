"""
Data Loading Utilities

Functions for loading and initial validation of the Diabetes dataset.
"""

import os
import pandas as pd
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_raw_data(filename="diabetic_data.csv"):
    """
    Load the raw Diabetes 130-US Hospitals dataset.
    
    Parameters
    ----------
    filename : str
        Name of the CSV file in the data/raw/ directory
        
    Returns
    -------
    pd.DataFrame
        Raw dataset
        
    Raises
    ------
    FileNotFoundError
        If the data file is not found
    """
    data_path = get_project_root() / "data" / "raw" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please download the dataset from UCI ML Repository and "
            f"place it in the data/raw/ directory."
        )
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    return df


def load_processed_data(filename="processed_data.csv"):
    """
    Load preprocessed data.
    
    Parameters
    ----------
    filename : str
        Name of the processed CSV file
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataset
    """
    data_path = get_project_root() / "data" / "processed" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            f"Please run the preprocessing notebook first."
        )
    
    print(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    
    return df


def save_processed_data(df, filename="processed_data.csv"):
    """
    Save preprocessed data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    filename : str
        Output filename
    """
    output_path = get_project_root() / "data" / "processed" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
