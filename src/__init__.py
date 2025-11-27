"""
Ontology-Aware Anomaly Detection Package

This package provides tools for anomaly detection with ontological constraints
on the Diabetes 130-US Hospitals dataset.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .preprocessing import load_raw_data, build_feature_matrix, train_test_split_stratified
from .models import IsolationForestDetector, AutoencoderDetector
from .ontology import compute_ontology_penalty, combine_scores
from .evaluation import compute_classification_metrics, plot_roc_pr_curves

__all__ = [
    "load_raw_data",
    "build_feature_matrix",
    "train_test_split_stratified",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "compute_ontology_penalty",
    "combine_scores",
    "compute_classification_metrics",
    "plot_roc_pr_curves",
]
