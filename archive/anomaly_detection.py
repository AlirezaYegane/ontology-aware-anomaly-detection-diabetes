"""
Anomaly Detection Module

Implementation of various anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Wrapper class for multiple anomaly detection algorithms.
    """
    
    def __init__(self, method='isolation_forest', contamination=0.1, **kwargs):
        """
        Initialize anomaly detector.
        
        Parameters
        ----------
        method : str
            Detection method: 'isolation_forest', 'lof', or 'ocsvm'
        contamination : float
            Expected proportion of outliers (0 to 0.5)
        **kwargs
            Additional parameters for the chosen method
        """
        self.method = method
        self.contamination = contamination
        self.model = self._initialize_model(**kwargs)
        
    def _initialize_model(self, **kwargs):
        """Initialize the chosen anomaly detection model."""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                **kwargs
            )
        elif self.method == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                **kwargs
            )
        elif self.method == 'ocsvm':
            return OneClassSVM(
                nu=self.contamination,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, X):
        """
        Fit the anomaly detection model.
        
        Parameters
        ----------
        X : array-like
            Training data
            
        Returns
        -------
        self
        """
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Parameters
        ----------
        X : array-like
            Data to predict
            
        Returns
        -------
        np.ndarray
            Predictions: 1 for normal, -1 for anomaly
        """
        return self.model.predict(X)
    
    def score_samples(self, X):
        """
        Compute anomaly scores.
        
        Parameters
        ----------
        X : array-like
            Data to score
            
        Returns
        -------
        np.ndarray
            Anomaly scores (lower = more anomalous)
        """
        if hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            raise NotImplementedError(f"Scoring not available for {self.method}")


def detect_anomalies(df, feature_cols=None, method='isolation_forest', 
                     contamination=0.1, **kwargs):
    """
    Detect anomalies in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list, optional
        Columns to use for detection. If None, use all numeric columns.
    method : str
        Detection method
    contamination : float
        Expected proportion of outliers
    **kwargs
        Additional parameters for the detector
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with anomaly predictions and scores
    AnomalyDetector
        Fitted detector object
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_cols].values
    
    # Initialize and fit detector
    detector = AnomalyDetector(method=method, contamination=contamination, **kwargs)
    detector.fit(X)
    
    # Get predictions and scores
    predictions = detector.predict(X)
    scores = detector.score_samples(X)
    
    # Add to dataframe
    result_df = df.copy()
    result_df['anomaly_prediction'] = predictions
    result_df['anomaly_score'] = scores
    result_df['is_anomaly'] = predictions == -1
    
    n_anomalies = (predictions == -1).sum()
    print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
    
    return result_df, detector


def compare_methods(df, feature_cols=None, contamination=0.1):
    """
    Compare multiple anomaly detection methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list, optional
        Columns to use for detection
    contamination : float
        Expected proportion of outliers
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    methods = ['isolation_forest', 'lof', 'ocsvm']
    results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} ---")
        result_df, detector = detect_anomalies(
            df, feature_cols=feature_cols, 
            method=method, contamination=contamination
        )
        results[method] = {
            'predictions': result_df['is_anomaly'],
            'scores': result_df['anomaly_score']
        }
    
    return results
