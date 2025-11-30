"""
Evaluation utilities for anomaly detection models.

This module provides:
- ROC-AUC / PR-AUC computation
- Threshold-based summaries
- Pretty-printed evaluation tables
- Single-model and multi-model ROC / PR plots
- Helpers for exporting metrics to disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "compute_roc_pr",
    "summarize_at_thresholds",
    "evaluate_anomaly_detector",
    "compute_classification_metrics",
    "plot_roc_pr_curves",
    "plot_evaluation_curves",
    "plot_roc_pr",
    "print_comparison_table",
    "save_metrics_summary",
]


# ============================================================================
# Core metric computation
# ============================================================================

def compute_roc_pr(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC metrics.

    Parameters
    ----------
    y_true :
        Binary ground-truth labels (0 = normal, 1 = anomaly).
    anomaly_scores :
        Anomaly scores from the model (higher = more suspicious).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "roc_auc"
        - "pr_auc"
    """
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def summarize_at_thresholds(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    percentiles: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    """
    Summarize precision/recall behavior at score percentiles.

    Parameters
    ----------
    y_true :
        Binary ground-truth labels.
    anomaly_scores :
        Anomaly scores from the model.
    percentiles :
        List of percentiles to evaluate (e.g. [90, 95, 99]).
        If None, defaults to [90, 95, 99].

    Returns
    -------
    List[Dict[str, float]]
        One dictionary per percentile with keys:
        - "percentile", "threshold", "precision", "recall", "flagged_pct"
    """
    if percentiles is None:
        percentiles = [90, 95, 99]

    results: List[Dict[str, float]] = []
    for percentile in percentiles:
        threshold = float(np.percentile(anomaly_scores, percentile))
        y_pred = (anomaly_scores >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        flagged_pct = (y_pred.sum() / len(y_pred)) * 100.0

        results.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "flagged_pct": float(flagged_pct),
            }
        )

    return results


# ============================================================================
# High-level evaluation with pretty printing
# ============================================================================

def evaluate_anomaly_detector(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluate an anomaly detector and print a compact console report.

    This is the verbose helper used in notebooks / scripts:
    - computes ROC-AUC and PR-AUC
    - prints a 90th/95th/99th percentile threshold table

    Parameters
    ----------
    y_true :
        Binary ground-truth labels.
    anomaly_scores :
        Anomaly scores from the model.
    model_name :
        Name of the model, used in printed headers.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "roc_auc"
        - "pr_auc"
        - "model_name"
    """
    roc_pr = compute_roc_pr(y_true, anomaly_scores)
    thresholds_summary = summarize_at_thresholds(y_true, anomaly_scores)

    roc_auc = roc_pr["roc_auc"]
    pr_auc = roc_pr["pr_auc"]

    print("\n" + "=" * 60)
    print(f"{model_name} - Evaluation Results")
    print("=" * 60)
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print("\n" + "-" * 60)
    print("Performance at Different Thresholds:")
    print("-" * 60)
    print(f"{'Percentile':<12} {'Precision':<12} {'Recall':<12} {'Flagged %':<12}")
    print("-" * 60)

    for row in thresholds_summary:
        p = row["percentile"]
        prec = row["precision"]
        rec = row["recall"]
        flagged = row["flagged_pct"]
        print(f"{p:>3}th        {prec:>8.4f}     {rec:>8.4f}     {flagged:>8.2f}%")

    print("=" * 60 + "\n")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "model_name": model_name,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Convenience wrapper used by ``run_pipeline_direct.py``.

    Computes ROC-AUC and PR-AUC and returns a metrics dict with:
        - ``roc_auc``
        - ``pr_auc``
        - ``model_name``

    Internally delegates to :func:`evaluate_anomaly_detector` so that
    notebooks and scripts share the same output format.
    """
    return evaluate_anomaly_detector(
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        model_name=model_name,
    )


# ============================================================================
# Plotting utilities
# ============================================================================

def plot_roc_pr_curves(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str = "Model",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot ROC and Precision–Recall curves for a single model.

    Parameters
    ----------
    y_true :
        Binary ground-truth labels (0/1).
    anomaly_scores :
        Anomaly scores (higher = more anomalous).
    title :
        Plot title prefix.
    save_path :
        Optional path to save the figure as PNG.
    show :
        If True, display the figure; otherwise it is closed after saving.
    """
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {title} (AUC = {roc_auc:.3f})")
    ax.grid(True, alpha=0.3)

    # Precision–Recall curve
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

    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved ROC/PR figure for '{title}' to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_evaluation_curves(
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
) -> None:
    """
    Plot ROC and Precision–Recall curves for multiple anomaly detectors.

    Parameters
    ----------
    y_true :
        Binary ground-truth labels.
    model_scores :
        Mapping from model name to anomaly score array.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    # ROC curves
    ax = axes[0]
    for i, (name, scores) in enumerate(model_scores.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        color = colors[i % len(colors)]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            linewidth=2,
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Anomaly Detection")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Precision–Recall curves
    ax = axes[1]
    for i, (name, scores) in enumerate(model_scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        color = colors[i % len(colors)]
        ax.plot(
            recall,
            precision,
            label=f"{name} (AUC = {pr_auc:.3f})",
            linewidth=2,
            color=color,
        )

    baseline = y_true.sum() / len(y_true)
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.3,
        label=f"Baseline ({baseline:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve - Anomaly Detection")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_roc_pr(
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
) -> None:
    """
    Alias for :func:`plot_evaluation_curves` kept for backward compatibility.
    """
    return plot_evaluation_curves(y_true, model_scores)


# ============================================================================
# Reporting helpers
# ============================================================================

def print_comparison_table(metrics_list: List[Dict[str, float]]) -> None:
    """
    Print a plain-text comparison table for multiple models.

    Parameters
    ----------
    metrics_list :
        List of metric dictionaries, each containing at least
        "model_name", "roc_auc", and "pr_auc".
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':<15} {'PR-AUC':<15}")
    print("-" * 60)

    for metrics in metrics_list:
        model_name = metrics["model_name"]
        roc_auc = metrics["roc_auc"]
        pr_auc = metrics["pr_auc"]
        print(f"{model_name:<25} {roc_auc:<15.4f} {pr_auc:<15.4f}")

    print("=" * 60 + "\n")


def save_metrics_summary(
    metrics_list: List[Dict[str, float]],
    output_path: str,
    format: str = "csv",
) -> None:
    """
    Save model comparison metrics to disk.

    Parameters
    ----------
    metrics_list :
        List of metric dictionaries.
    output_path :
        Destination file path.
    format :
        Output format, one of {"csv", "json"}.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df = pd.DataFrame(metrics_list)
        df.to_csv(path, index=False)
        print(f"Metrics saved to: {path}")
    elif format == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics_list, f, indent=2)
        print(f"Metrics saved to: {path}")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
