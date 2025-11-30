"""
Top-level package for the Ontology-Aware Anomaly Detection project.

This module exposes the main public API used by scripts and notebooks:
configuration, logging helpers, preprocessing utilities, anomaly models,
ontology integration, and evaluation helpers.
"""

from .config import GLOBAL_CONFIG
from .logger import get_logger
from .preprocessing import (
    load_raw_data,
    build_feature_matrix,
    train_test_split_stratified,
    get_selected_features,
    clean_data,
    create_target,
)
from .models import (
    IsolationForestDetector,
    AutoencoderDetector,
    DecisionTreeDetector,
    RandomForestDetector,
)
from .ontology import apply_ontology_rules, combine_scores
from .evaluation import compute_classification_metrics, plot_roc_pr_curves

__all__ = [
    # config & logging
    "GLOBAL_CONFIG",
    "get_logger",
    # preprocessing
    "load_raw_data",
    "build_feature_matrix",
    "train_test_split_stratified",
    "get_selected_features",
    "clean_data",
    "create_target",
    # models
    "IsolationForestDetector",
    "AutoencoderDetector",
    "DecisionTreeDetector",
    "RandomForestDetector",
    # ontology layer
    "apply_ontology_rules",
    "combine_scores",
    # evaluation
    "compute_classification_metrics",
    "plot_roc_pr_curves",
]

__version__ = "0.1.0"
__author__ = "Alireza Yegane"
