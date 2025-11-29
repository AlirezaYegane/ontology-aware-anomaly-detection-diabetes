# src/config.py
from dataclasses import dataclass, field
from typing import List


# =========================
# Data-related configuration
# =========================
@dataclass
class DataConfig:
    # نسبت تست‌ست
    test_size: float = 0.2
    # برای اسپلیت‌های مختلف در Step 5
    random_seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 2025]
    )


# =========================
# Isolation Forest config
# =========================
@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    # اگر < 0.0 باشد، از train_pos_rate استفاده می‌کنیم
    contamination: float = -1.0
    random_state: int = 42


# =========================
# Autoencoder config
# =========================
@dataclass
class AutoencoderConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3


# =========================
# Ontology / rules config
# (فعلاً تو run_pipeline مستقیم λها رو نوشتی،
#  ولی این structure رو هم داریم برای آینده)
# =========================
@dataclass
class OntologyConfig:
    lambda_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.3, 0.5]
    )


# =========================
# Evaluation (multi-split) config
# (اگر بعداً خواستی seeds رو از اینجا بخونی)
# =========================
@dataclass
class EvaluationConfig:
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 2025]
    )


# =========================
# Global config object
# =========================
@dataclass
class GlobalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    autoencoder: AutoencoderConfig = field(
        default_factory=AutoencoderConfig
    )
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig
    )


GLOBAL_CONFIG = GlobalConfig()
