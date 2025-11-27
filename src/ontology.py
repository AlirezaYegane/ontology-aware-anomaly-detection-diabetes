"""
Ontology-Inspired Rule Layer for Anomaly Detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def compute_ontology_penalty(row: pd.Series) -> float:
    """
    Compute a domain-knowledge-based penalty score for a patient record.
    
    Rules:
    1. HIGH RISK (penalty = 0.9):
       - Poor glycemic control (A1Cresult > 7 or > 8) AND
       - No medication changes AND
       - Patient is on diabetes medication
       
    2. HIGH RISK (penalty = 0.85):
       - Very high glucose levels (max_glu_serum > 200 or > 300) AND
       - Few lab procedures (< 40)
       
    3. MEDIUM RISK (penalty = 0.6):
       - High medication burden (num_medications > 20) AND
       - Short hospital stay (time_in_hospital < 3 days)
       
    4. LOW RISK (penalty = 0.1):
       - None of the above conditions met
    """
    penalty = 0.1  # Default low penalty
    
    # Rule 1: Poor glycemic control + no medication adjustment + on diabetes meds
    if ('A1Cresult' in row.index and 
        row.get('A1Cresult', '') in ['>7', '>8'] and
        row.get('change', '') == 'No' and
        row.get('diabetesMed', '') == 'Yes'):
        penalty = max(penalty, 0.9)
    
    # Rule 2: Very high glucose + insufficient lab monitoring
    if ('max_glu_serum' in row.index and 
        row.get('max_glu_serum', '') in ['>200', '>300'] and
        row.get('num_lab_procedures', 999) < 40):
        penalty = max(penalty, 0.85)
    
    # Rule 3: High medication burden + short stay
    if (row.get('num_medications', 0) > 20 and
        row.get('time_in_hospital', 999) < 3):
        penalty = max(penalty, 0.6)
    
    return penalty

def apply_ontology_rules(df: pd.DataFrame) -> pd.Series:
    """Apply ontology rules to the entire DataFrame."""
    return df.apply(compute_ontology_penalty, axis=1)

def combine_scores(ml_scores: np.ndarray, ontology_penalties: np.ndarray, 
                  alpha: float = 0.7, beta: float = 0.3) -> np.ndarray:
    """
    Combine ML anomaly scores with ontology penalties.
    
    The ML scores are normalized to [0, 1] range before combining to ensure
    both components are on a comparable scale.
    
    Final Score = alpha * normalized(ML_Score) + beta * Ontology_Penalty
    
    Args:
        ml_scores: Raw anomaly scores from ML model
        ontology_penalties: Penalty scores from ontology rules (assumed 0-1 range)
        alpha: Weight for ML score component (default: 0.7)
        beta: Weight for ontology penalty component (default: 0.3)
        
    Returns:
        Combined anomaly scores
    """
    # Normalize ML scores to [0, 1] range
    ml_min, ml_max = ml_scores.min(), ml_scores.max()
    if ml_max > ml_min:
        ml_scores_normalized = (ml_scores - ml_min) / (ml_max - ml_min)
    else:
        ml_scores_normalized = np.zeros_like(ml_scores)
    
    # Combine normalized scores
    return alpha * ml_scores_normalized + beta * ontology_penalties
