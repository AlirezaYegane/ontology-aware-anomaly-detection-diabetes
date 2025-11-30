# src/config.py
from dataclasses import dataclass, field
from typing import List


# =========================
# Data-related configuration
# =========================
@dataclass
class DataConfig:
    """Configuration for data splitting and sampling."""
    # Fraction of the dataset used as test set
    test_size: float = 0.2
    # Random seeds used for different splits (single-run + multi-split eval)
    random_seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 2025]
    )


# =========================
# Isolation Forest config
# =========================
@dataclass
class IsolationForestConfig:
    """Hyperparameters for the Isolation Forest detector."""
    n_estimators: int = 200
    # If < 0.0, the training positive rate is used as contamination
    contamination: float = -1.0
    random_state: int = 42


# =========================
# Autoencoder config
# =========================
@dataclass
class AutoencoderConfig:
    """Hyperparameters for the Autoencoder-based detector."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3


# =========================
# Ontology / rules config
# =========================
@dataclass
class OntologyConfig:
    """
    Configuration for ontology integration.

    lambda_grid controls the weight given to ontology penalties when
    combining model scores with rule-based signals.
    """
    lambda_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.3, 0.5]
    )


# =========================
# Global config object
# =========================
@dataclass
class GlobalConfig:
    """Top-level container for all configuration sections."""
    data: DataConfig = field(default_factory=DataConfig)
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    autoencoder: AutoencoderConfig = field(
        default_factory=AutoencoderConfig
    )
    ontology: OntologyConfig = field(default_factory=OntologyConfig)


# Singleton-style global configuration used across the project
GLOBAL_CONFIG = GlobalConfig()
