"""
Evaluation module for Anomaly Detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score
)
from typing import Dict, List, Optional
import json
from pathlib import Path

def evaluate_anomaly_detector(y_true: np.ndarray, anomaly_scores: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate anomaly detector using various metrics.
    """
    # Compute ROC-AUC
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    
    # Compute PR-AUC
    pr_auc = average_precision_score(y_true, anomaly_scores)
    
    # Example thresholds: percentiles of anomaly scores
    thresholds = [
        np.percentile(anomaly_scores, 90),
        np.percentile(anomaly_scores, 95),
        np.percentile(anomaly_scores, 99)
    ]
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print(f"\n{'-'*60}")
    print(f"Performance at Different Thresholds:")
    print(f"{'-'*60}")
    print(f"{'Percentile':<12} {'Precision':<12} {'Recall':<12} {'Flagged %':<12}")
    print(f"{'-'*60}")
    
    for percentile, threshold in zip([90, 95, 99], thresholds):
        y_pred = (anomaly_scores >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        flagged_pct = (y_pred.sum() / len(y_pred)) * 100
        
        print(f"{percentile:>3}th        {precision:>8.4f}     {recall:>8.4f}     {flagged_pct:>8.2f}%")
    
    print(f"{'='*60}\n")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'model_name': model_name
    }

def compute_roc_pr(y_true: np.ndarray, anomaly_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC metrics.
    Specification-compliant function for metric computation.
    
    Args:
        y_true: True labels
        anomaly_scores: Anomaly scores from model
        
    Returns:
        Dictionary with 'roc_auc' and 'pr_auc' keys
    """
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

def summarize_at_thresholds(y_true: np.ndarray, anomaly_scores: np.ndarray, 
                            percentiles: List[int] = [90, 95, 99]) -> List[Dict[str, float]]:
    """
    Summarize performance at different threshold percentiles.
    Specification-compliant function for threshold analysis.
    
    Args:
        y_true: True labels
        anomaly_scores: Anomaly scores from model
        percentiles: List of percentile values to evaluate
        
    Returns:
        List of dictionaries containing threshold metrics
    """
    results = []
    for percentile in percentiles:
        threshold = np.percentile(anomaly_scores, percentile)
        y_pred = (anomaly_scores >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        flagged_pct = (y_pred.sum() / len(y_pred)) * 100
        
        results.append({
            'percentile': percentile,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'flagged_pct': flagged_pct
        })
    
    return results

def plot_evaluation_curves(y_test: np.ndarray, model_scores: Dict[str, np.ndarray]):
    """
    Plot ROC and Precision-Recall curves for multiple anomaly detectors.
    
    Args:
        y_test: True labels
        model_scores: Dictionary mapping model names to their anomaly scores
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # === ROC Curve ===
    ax = axes[0]
    
    for i, (name, scores) in enumerate(model_scores.items()):
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', 
                linewidth=2, color=color)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # === Precision-Recall Curve ===
    ax = axes[1]
    
    for i, (name, scores) in enumerate(model_scores.items()):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        color = colors[i % len(colors)]
        ax.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', 
                linewidth=2, color=color)
    
    # Baseline (proportion of positives)
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
               alpha=0.3, label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_comparison_table(metrics_list: List[Dict[str, float]]):
    """
    Print a comparison table of multiple models.
    """
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'ROC-AUC':<15} {'PR-AUC':<15}")
    print(f"{'-'*60}")
    
    for metrics in metrics_list:
        model_name = metrics['model_name']
        roc_auc = metrics['roc_auc']
        pr_auc = metrics['pr_auc']
        print(f"{model_name:<25} {roc_auc:<15.4f} {pr_auc:<15.4f}")
    
    print(f"{'='*60}\n")

def plot_roc_pr(y_test: np.ndarray, model_scores: Dict[str, np.ndarray]):
    """
    Plot ROC and PR curves for multiple models.
    Alias for plot_evaluation_curves() - specification-compliant name.
    """
    return plot_evaluation_curves(y_test, model_scores)

def save_metrics_summary(metrics_list: List[Dict[str, float]], output_path: str, format: str = 'csv'):
    """
    Save model comparison metrics to a file.
    Specification-compliant function for exporting results.
    
    Args:
        metrics_list: List of metric dictionaries
        output_path: Path to save the summary file
        format: Output format - 'csv' or 'json' (default: 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(metrics_list)
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to: {output_path}")
    elif format == 'json':
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        print(f"Metrics saved to: {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
def compute_classification_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Wrapper used by run_pipeline_direct.py.

    Computes ROC-AUC and PR-AUC and returns a metrics dict with:
        - 'roc_auc'
        - 'pr_auc'
        - 'model_name'

    Internally از evaluate_anomaly_detector استفاده می‌کند
    تا رفتار در نوت‌بوک‌ها و اسکریپت یکی باشد.
    """
    return evaluate_anomaly_detector(
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        model_name=model_name,
    )


def plot_roc_pr_curves(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str = "Model",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot ROC and Precision–Recall curves for a SINGLE model,
    و نتیجه را اگر save_path داده شده باشد روی دیسک ذخیره می‌کند.

    Parameters
    ----------
    y_true : np.ndarray
        برچسب‌های باینری واقعی (0/1).
    anomaly_scores : np.ndarray
        نمره‌ی آنومالی، هرچه بیشتر = مشکوک‌تر.
    title : str
        عنوان نمودارها.
    save_path : Optional[str]
        مسیر ذخیره‌ی فایل PNG.
    show : bool
        اگر True باشد نمودار نمایش داده می‌شود، در غیر این صورت فقط ذخیره/بسته می‌شود.
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    # PR
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- ROC Curve ---
    ax = axes[0]
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {title} (AUC = {roc_auc:.3f})")
    ax.grid(True, alpha=0.3)

    # --- Precision–Recall Curve ---
    ax = axes[1]
    ax.plot(recall, precision, linewidth=2)
    baseline = y_true.sum() / len(y_true) if len(y_true) > 0 else 0.0
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.3,
        label=f"Baseline = {baseline:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — {title} (AUC = {pr_auc:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    # ذخیره‌ی فایل اگر مسیر داده شده
    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved ROC/PR figure for '{title}' to: {save_path}")

    # نمایش یا بستن شکل
    if show:
        plt.show()
    else:
        plt.close(fig)

