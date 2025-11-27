"""
Ontology-Inspired Rule Layer for Anomaly Detection
===================================================

This module implements a domain-knowledge-based penalty system to enhance
anomaly detection for early hospital readmission prediction. It combines
machine learning anomaly scores (IsolationForest) with expert rule-based
penalties derived from clinical domain knowledge.

Author: ML Research Team
Dataset: Diabetes 130-US Hospitals (1999-2008)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def compute_ontology_penalty(row):
    """
    Compute a domain-knowledge-based penalty score for a patient record.
    
    This function implements clinical rules that identify potentially risky
    patient scenarios based on combinations of test results, medications,
    and care patterns.
    
    Parameters:
    -----------
    row : pd.Series
        A single patient record containing relevant clinical features
        
    Returns:
    --------
    float
        Penalty score between 0 and 1, where higher values indicate
        higher risk according to ontology-based rules
        
    Rules:
    ------
    1. HIGH RISK (penalty = 0.9):
       - Poor glycemic control (A1Cresult > 7 or > 8) AND
       - No medication changes AND
       - Patient is on diabetes medication
       → Indicates potential treatment non-compliance or inadequate management
       
    2. HIGH RISK (penalty = 0.85):
       - Very high glucose levels (max_glu_serum > 200 or > 300) AND
       - Few lab procedures (< 40)
       → Suggests insufficient monitoring of severe hyperglycemia
       
    3. MEDIUM RISK (penalty = 0.6):
       - High medication burden (num_medications > 20) AND
       - Short hospital stay (time_in_hospital < 3 days)
       → May indicate complex case with inadequate stabilization
       
    4. LOW RISK (penalty = 0.1):
       - None of the above conditions met
    """
    penalty = 0.1  # Default low penalty
    
    # Rule 1: Poor glycemic control + no medication adjustment + on diabetes meds
    # This is a red flag for treatment non-adherence or inadequate therapy
    if ('A1Cresult' in row.index and 
        row.get('A1Cresult', '') in ['>7', '>8'] and
        row.get('change', '') == 'No' and
        row.get('diabetesMed', '') == 'Yes'):
        penalty = max(penalty, 0.9)
    
    # Rule 2: Very high glucose + insufficient lab monitoring
    # Severe hyperglycemia requires close monitoring
    if ('max_glu_serum' in row.index and 
        row.get('max_glu_serum', '') in ['>200', '>300'] and
        row.get('num_lab_procedures', 999) < 40):
        penalty = max(penalty, 0.85)
    
    # Rule 3: High medication burden + short stay
    # Complex patients may need longer stabilization
    if (row.get('num_medications', 0) > 20 and
        row.get('time_in_hospital', 999) < 3):
        penalty = max(penalty, 0.6)
    
    return penalty


def evaluate_anomaly_scores(y_true, scores, score_name="Score"):
    """
    Evaluate anomaly detection performance using ROC-AUC and PR-AUC.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 = positive class, 0 = negative class)
    scores : array-like
        Anomaly scores (higher = more anomalous)
    score_name : str
        Name of the scoring method for display purposes
        
    Returns:
    --------
    dict
        Dictionary containing roc_auc and pr_auc metrics
    """
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    
    return {
        'score_name': score_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def compare_anomaly_methods(df_test):
    """
    Apply ontology-based rules and compare performance with baseline IsolationForest.
    
    This function:
    1. Computes ontology penalty for each record
    2. Creates a combined anomaly score
    3. Evaluates both baseline and combined approaches
    4. Prints comparison results
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Test dataset containing:
        - Original feature columns from Diabetes dataset
        - 'iforest_score': normalized IsolationForest anomaly scores
        - 'y': binary label (1 = early readmission, 0 = otherwise)
        
    Returns:
    --------
    pd.DataFrame
        Test dataframe with added 'ontology_penalty' and 'final_score' columns
    """
    
    print("=" * 70)
    print("ONTOLOGY-INSPIRED ANOMALY DETECTION EVALUATION")
    print("=" * 70)
    print()
    
    # Step 1: Compute ontology penalty for each row
    print("Step 1: Computing ontology-based penalties...")
    df_test['ontology_penalty'] = df_test.apply(compute_ontology_penalty, axis=1)
    print(f"✓ Computed penalties for {len(df_test)} records")
    print(f"  Penalty range: [{df_test['ontology_penalty'].min():.3f}, "
          f"{df_test['ontology_penalty'].max():.3f}]")
    print(f"  Penalty mean: {df_test['ontology_penalty'].mean():.3f}")
    print()
    
    # Step 2: Create combined anomaly score
    print("Step 2: Creating combined anomaly score...")
    alpha = 0.7  # Weight for IsolationForest score
    beta = 0.3   # Weight for ontology penalty
    df_test['final_score'] = (alpha * df_test['iforest_score'] + 
                              beta * df_test['ontology_penalty'])
    print(f"✓ Combined score: {alpha:.1f} × IForest + {beta:.1f} × Ontology")
    print()
    
    # Step 3: Evaluate both methods
    print("Step 3: Evaluating anomaly detection performance...")
    print()
    
    # Baseline: IsolationForest only
    baseline_metrics = evaluate_anomaly_scores(
        df_test['y'], 
        df_test['iforest_score'],
        "IsolationForest (Baseline)"
    )
    
    # Enhanced: Combined score
    combined_metrics = evaluate_anomaly_scores(
        df_test['y'],
        df_test['final_score'],
        "Combined (IForest + Ontology)"
    )
    
    # Step 4: Display comparison table
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Method':<35} {'ROC-AUC':<12} {'PR-AUC':<12}")
    print("-" * 70)
    print(f"{baseline_metrics['score_name']:<35} "
          f"{baseline_metrics['roc_auc']:<12.4f} "
          f"{baseline_metrics['pr_auc']:<12.4f}")
    print(f"{combined_metrics['score_name']:<35} "
          f"{combined_metrics['roc_auc']:<12.4f} "
          f"{combined_metrics['pr_auc']:<12.4f}")
    print("-" * 70)
    
    # Calculate improvements
    roc_improvement = combined_metrics['roc_auc'] - baseline_metrics['roc_auc']
    pr_improvement = combined_metrics['pr_auc'] - baseline_metrics['pr_auc']
    
    print(f"{'Δ Improvement':<35} "
          f"{roc_improvement:+<12.4f} "
          f"{pr_improvement:+<12.4f}")
    print("=" * 70)
    print()
    
    # Step 5: Textual summary
    print("SUMMARY")
    print("=" * 70)
    
    if roc_improvement > 0 and pr_improvement > 0:
        print("✓ The ontology-inspired rule layer IMPROVED performance:")
        print(f"  • ROC-AUC increased by {roc_improvement:.4f} "
              f"({roc_improvement/baseline_metrics['roc_auc']*100:+.2f}%)")
        print(f"  • PR-AUC increased by {pr_improvement:.4f} "
              f"({pr_improvement/baseline_metrics['pr_auc']*100:+.2f}%)")
        print()
        print("  Interpretation: Domain-knowledge-based rules successfully")
        print("  identified additional risk patterns that pure statistical")
        print("  anomaly detection missed, particularly around medication")
        print("  management and glycemic control.")
        
    elif roc_improvement < 0 or pr_improvement < 0:
        print("⚠ The ontology-inspired rule layer DECREASED performance:")
        print(f"  • ROC-AUC changed by {roc_improvement:.4f} "
              f"({roc_improvement/baseline_metrics['roc_auc']*100:+.2f}%)")
        print(f"  • PR-AUC changed by {pr_improvement:.4f} "
              f"({pr_improvement/baseline_metrics['pr_auc']*100:+.2f}%)")
        print()
        print("  Interpretation: The current rule set may be too simplistic")
        print("  or not aligned with actual readmission patterns. Consider:")
        print("  - Refining rule thresholds through expert consultation")
        print("  - Adding more nuanced interaction patterns")
        print("  - Adjusting the weighting (α, β) between ML and rules")
        
    else:
        print("→ The ontology-inspired rule layer had MINIMAL impact:")
        print(f"  • ROC-AUC changed by {roc_improvement:.4f}")
        print(f"  • PR-AUC changed by {pr_improvement:.4f}")
        print()
        print("  Interpretation: Rules may be redundant with ML-detected")
        print("  patterns, or the weighting needs adjustment.")
    
    print("=" * 70)
    print()
    
    # Distribution insights
    print("ANOMALY SCORE DISTRIBUTIONS")
    print("=" * 70)
    high_ontology_penalty = (df_test['ontology_penalty'] > 0.5).sum()
    print(f"Records with high ontology penalty (>0.5): {high_ontology_penalty} "
          f"({high_ontology_penalty/len(df_test)*100:.1f}%)")
    
    high_iforest = (df_test['iforest_score'] > df_test['iforest_score'].median()).sum()
    print(f"Records with high IForest score (>median): {high_iforest} "
          f"({high_iforest/len(df_test)*100:.1f}%)")
    
    # Check overlap
    both_high = ((df_test['ontology_penalty'] > 0.5) & 
                 (df_test['iforest_score'] > df_test['iforest_score'].median())).sum()
    print(f"Records flagged by BOTH methods: {both_high} "
          f"({both_high/len(df_test)*100:.1f}%)")
    print("=" * 70)
    print()
    
    return df_test


# Example usage for Jupyter notebook
if __name__ == "__main__":
    """
    Example execution. In a Jupyter notebook, you would have df_test 
    already loaded from your data pipeline.
    """
    
    print("=" * 70)
    print("NOTE: This is example code for demonstration.")
    print("In your notebook, you should have df_test already prepared with:")
    print("  - Original Diabetes 130-US features")
    print("  - 'iforest_score' column (normalized)")
    print("  - 'y' column (binary label)")
    print("=" * 70)
    print()
    print("Example notebook usage:")
    print()
    print("```python")
    print("# After loading and preparing your test data...")
    print("from ontology_rule_layer import compare_anomaly_methods")
    print()
    print("# Apply ontology rules and evaluate")
    print("df_test_enhanced = compare_anomaly_methods(df_test)")
    print()
    print("# Inspect results")
    print("df_test_enhanced[['iforest_score', 'ontology_penalty', 'final_score', 'y']].head(10)")
    print("```")
    print()
