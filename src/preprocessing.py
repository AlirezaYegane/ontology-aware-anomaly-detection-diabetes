"""
Preprocessing module for the Diabetes Hospital Readmission dataset.

This module implements a transparent, instrumented preprocessing pipeline
that turns the raw CSV file into a clean feature matrix and binary target.

Key responsibilities
--------------------
- Column selection (small, clinically meaningful feature set)
- Missing value handling (explicit, conservative dropping on critical fields)
- Simple, interpretable imputation (median / 'Unknown')
- One-hot encoding and scaling
- Filter tracking and summary report generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pickle

from .filter_tracker import FilterTracker


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.

    Alias for load_data() – kept for API compatibility with scripts/notebooks.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    return load_data(filepath)


# -------------------------------------------------------------------------
# Feature selection
# -------------------------------------------------------------------------


def get_selected_features() -> List[str]:
    """
    Return the list of clinically meaningful features used by the models.

    The full dataset contains ~50 columns; here we restrict the feature space
    to a small, interpretable subset that captures demographics, utilisation,
    medication intensity, and glycaemic control.
    """
    return [
        # Demographics
        "race",
        "gender",
        "age",
        # Hospital stay metrics
        "time_in_hospital",
        # Procedure / utilisation counts
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_inpatient",
        "number_emergency",
        # Lab results
        "A1Cresult",
        "max_glu_serum",
        # Medication changes
        "change",
        "diabetesMed",
    ]


# -------------------------------------------------------------------------
# Cleaning & target construction
# -------------------------------------------------------------------------


def clean_data(
    df: pd.DataFrame,
    selected_features: List[str],
    tracker: Optional[FilterTracker] = None,
) -> pd.DataFrame:
    """
    Clean the dataset with explicit, instrumented filtering.

    Steps
    -----
    1. Select specific features + target ("readmitted").
    2. Replace '?' markers with NaN.
    3. Drop rows with missing values in CRITICAL columns only.

    Filtering rationale
    -------------------
    - Feature selection:
        Reduces from ~50 raw columns to a focused set of 14 clinical features,
        chosen for interpretability and relevance to readmission risk.

    - Missing markers:
        The original dataset uses '?' as a missing-value marker; this is
        converted to NaN for consistent downstream handling.

    - Row removal:
        Rows are dropped ONLY when critical fields (target + key clinical
        indicators) are missing. Non-critical columns are retained and later
        imputed (numerical: median, categorical: 'Unknown').

        This avoids the earlier behaviour where a blanket dropna() removed
        almost the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    selected_features : List[str]
        Feature names to keep.
    tracker : FilterTracker, optional
        If provided, all filtering steps will be recorded.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Non-critical columns may still contain NaNs.
    """
    df_start = df.copy()

    # Step 1: select subset of columns (feature selection)
    df_subset = df[selected_features + ["readmitted"]].copy()

    if tracker:
        tracker.track_step(
            df_start,
            df_subset,
            step_name="feature_selection",
            description=(
                f"Select {len(selected_features)} features "
                f"(from {len(df.columns)} columns)"
            ),
        )

    # Step 2: replace '?' with NaN
    df_before_replace = df_subset.copy()
    df_clean = df_subset.replace("?", np.nan)

    if tracker:
        n_question_marks = (df_before_replace == "?").sum().sum()
        tracker.track_step(
            df_before_replace,
            df_clean,
            step_name="replace_missing_markers",
            description=f"Replace '?' with NaN ({n_question_marks:,} cells affected)",
        )

    # Step 3: drop rows with missing values in CRITICAL columns only
    critical_cols = [
        "readmitted",
        "race",
        "gender",
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "number_inpatient",
        "number_emergency",
        "change",
        "diabetesMed",
    ]
    critical_cols = [c for c in critical_cols if c in df_clean.columns]

    df_before_drop = df_clean.copy()
    df_clean = df_clean.dropna(subset=critical_cols)

    if tracker:
        n_removed = len(df_before_drop) - len(df_clean)
        tracker.track_step(
            df_before_drop,
            df_clean,
            step_name="drop_missing_values",
            description=(
                "Drop rows with missing values in critical columns "
                f"{critical_cols} ({n_removed:,} rows removed)"
            ),
        )

    return df_clean


def create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a binary target variable from the 'readmitted' column.

    Definition
    ----------
    y = 1  if readmitted == '<30'
    y = 0  otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame that still contains the 'readmitted' column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except 'readmitted').
    y : pd.Series
        Binary target.
    """
    y = (df["readmitted"] == "<30").astype(int)
    X = df.drop("readmitted", axis=1)
    return X, y


# -------------------------------------------------------------------------
# Encoding / scaling / imputation
# -------------------------------------------------------------------------


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """
    Extract feature names from the fitted ColumnTransformer.
    """
    feature_names: List[str] = []
    feature_names.extend(numerical_cols)

    cat_transformer = preprocessor.named_transformers_["cat"]
    if hasattr(cat_transformer, "get_feature_names_out"):
        cat_features = cat_transformer.get_feature_names_out(categorical_cols)
        feature_names.extend(list(cat_features))

    return feature_names


def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, transparent imputation for remaining missing values.

    - Numerical columns: fill NaN with the column median.
    - Categorical columns: fill NaN with the string 'Unknown'.
    """
    X_imputed = X.copy()

    # Numerical columns -> median
    num_cols = X_imputed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X_imputed[col].isna().any():
            median = X_imputed[col].median()
            X_imputed[col] = X_imputed[col].fillna(median)

    # Categorical columns -> 'Unknown'
    cat_cols = X_imputed.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if X_imputed[col].isna().any():
            X_imputed[col] = X_imputed[col].fillna("Unknown")

    return X_imputed


def fit_transform_data(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Apply one-hot encoding and standard scaling to the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix with mixed dtypes.

    Returns
    -------
    X_transformed : pd.DataFrame
        Fully numeric, encoded and scaled feature matrix.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols)

    X_final = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )
    return X_final, preprocessor


# -------------------------------------------------------------------------
# Full preprocessing pipeline
# -------------------------------------------------------------------------


def build_feature_matrix(
    data_or_path: Union[pd.DataFrame, str, Path],
    save_preprocessor: bool = False,
    output_dir: Optional[Path] = None,
    enable_tracking: bool = True,
    save_summary: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Optional[ColumnTransformer]]:
    """
    Full preprocessing pipeline from raw data to feature matrix.

    This function is the main entry point used by the pipeline script.

    Behaviour
    ---------
    - Accepts either a DataFrame or a CSV filepath.
    - Runs the clean_data() pipeline with FilterTracker instrumentation.
    - Constructs the binary target (readmitted < 30 days).
    - Imputes remaining missing values.
    - Applies encoding + scaling.
    - Optionally saves the preprocessor and filtering summaries.

    Parameters
    ----------
    data_or_path : pd.DataFrame or str or Path
        Raw data or path to the CSV file.
    save_preprocessor : bool, default False
        If True, persist the fitted preprocessor to disk.
    output_dir : Path, optional
        Base directory for saving artefacts (preprocessor, reports).
    enable_tracking : bool, default True
        Enable FilterTracker instrumentation.
    save_summary : bool, default True
        Save filtering summary reports (JSON + Markdown).

    Returns
    -------
    X_transformed : pd.DataFrame
        Encoded and scaled feature matrix.
    y : pd.Series
        Binary target (1 = readmitted < 30 days).
    preprocessor : ColumnTransformer or None
        Fitted preprocessor. None if something goes wrong unusually.
    """
    tracker = FilterTracker() if enable_tracking else None

    if tracker:
        print("\n" + "=" * 80)
        print("  DATA FILTERING & PREPROCESSING PIPELINE")
        print("=" * 80 + "\n")

    # Step 1: load or accept data
    if isinstance(data_or_path, pd.DataFrame):
        df = data_or_path.copy()
    else:
        df = load_raw_data(str(data_or_path))

    df_initial = df.copy()

    if tracker:
        tracker.track_step(
            df_initial,
            df_initial,
            step_name="load_raw_data",
            description=(
                f"Load raw dataset ({df.shape[0]:,} rows × {df.shape[1]} columns)"
            ),
        )

    # Step 2: select features
    selected_features = get_selected_features()

    # Step 3: clean data (with tracking)
    df_clean = clean_data(df, selected_features, tracker=tracker)

    # Step 4: create target variable
    X_before_target = df_clean.drop("readmitted", axis=1)
    X, y = create_target(df_clean)

    if tracker:
        tracker.track_step(
            X_before_target,
            X,
            step_name="create_target",
            description="Create binary target (readmitted <30 days)",
        )

    # Step 5: impute remaining missing values
    X_imputed = impute_missing_values(X)

    if tracker:
        tracker.track_step(
            X,
            X_imputed,
            step_name="impute_missing_values",
            description="Impute missing values (numerical: median, categorical: 'Unknown')",
        )

    # Step 6: fit and transform features
    X_transformed, preprocessor = fit_transform_data(X_imputed)

    if tracker:
        tracker.track_step(
            X_imputed,
            X_transformed,
            step_name="encode_and_scale",
            description=(
                "OneHotEncode + StandardScale "
                f"({X_imputed.shape[1]} -> {X_transformed.shape[1]} features)"
            ),
        )
        tracker.track_class_balance(y)

    # Optional: save preprocessor
    if save_preprocessor and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

    # Step 7: generate and save summary reports
    if tracker:
        tracker.print_summary_table()

        if save_summary:
            if output_dir is not None:
                reports_dir = Path(output_dir) / "reports"
            else:
                reports_dir = Path("results") / "reports"
            tracker.save_summary(reports_dir, basename="data_filtering_summary")

    return X_transformed, y, preprocessor


# -------------------------------------------------------------------------
# Train/test split & persistence helpers
# -------------------------------------------------------------------------


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train–test split.

    This is a thin wrapper around sklearn.model_selection.train_test_split,
    kept as a named function for clarity and testability.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def save_processed_data(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: Path,
) -> None:
    """
    Save the processed features, target, and preprocessor to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X.to_csv(output_dir / "X_features.csv", index=False)
    y.to_csv(output_dir / "y_target.csv", index=False, header=["target"])

    with open(output_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)


if __name__ == "__main__":
    # Minimal example entry point (kept for convenience / debugging).
    print("Running preprocessing example...")
    data_path = Path("data") / "raw" / "diabetic_data.csv"
    if data_path.exists():
        df_raw = load_raw_data(data_path)
        X, y, preprocessor = build_feature_matrix(df_raw, save_preprocessor=False)
        save_processed_data(X, y, preprocessor, Path("data") / "processed")
        print("Done.")
    else:
        print(f"File not found: {data_path}")
