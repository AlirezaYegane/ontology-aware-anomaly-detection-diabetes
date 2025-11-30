"""
Model definitions for hospital readmission anomaly detection.

Includes:
    - FeedforwardAutoencoder (PyTorch module)
    - IsolationForestDetector (unsupervised baseline)
    - AutoencoderDetector (unsupervised reconstruction baseline)
    - DecisionTreeDetector (supervised baseline)
    - RandomForestDetector (supervised baseline)
    - Thin helper aliases for spec-compatibility
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Feed-forward Autoencoder backbone
# =============================================================================


class FeedforwardAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder.

    Encoder:  input_dim -> h1 -> h2 -> ... -> bottleneck
    Decoder:  bottleneck -> ... -> h2 -> h1 -> input_dim
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers: list[nn.Module] = []
        rev_hidden = list(reversed(hidden_dims))
        prev_dim = rev_hidden[0]
        for h in rev_hidden[1:]:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# =============================================================================
# Isolation Forest Detector (unsupervised)
# =============================================================================


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 100,
) -> IsolationForest:
    """
    Train a basic Isolation Forest model on X_train.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    model.fit(X_train)
    return model


def get_if_anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Compute anomaly scores from an Isolation Forest model.

    sklearn's IsolationForest.score_samples:
        - higher score = more "normal"

    We convert it to an anomaly score:
        - higher score = more anomalous / suspicious
    by taking the negative of score_samples.
    """
    scores = -model.score_samples(X)
    return scores


class IsolationForestDetector:
    """
    Wrapper used by run_pipeline_direct.py.

    Methods:
        - fit(X_train)
        - predict_scores(X) -> anomaly scores (higher = more anomalous)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model: Optional[IsolationForest] = None

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        self.model = train_isolation_forest(
            X_train,
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector is not fitted. Call fit() first.")
        return get_if_anomaly_scores(self.model, X)


# =============================================================================
# Autoencoder Detector (unsupervised)
# =============================================================================


class AutoencoderDetector:
    """
    High-level detector API for the feed-forward autoencoder.

    Typical usage:

        ae = AutoencoderDetector(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32],
            epochs=50,
            batch_size=256,
            learning_rate=1e-3,
        )
        ae.fit(X_train_normal)
        scores = ae.predict_scores(X_test)

    The anomaly score for each sample is its mean-squared reconstruction error.
    Higher scores correspond to more anomalous samples.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = FeedforwardAutoencoder(input_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # mean squared error per feature
        self.criterion = nn.MSELoss(reduction="none")

    def _to_numpy(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return X.astype(np.float32)

    def fit(self, X_train: pd.DataFrame | np.ndarray) -> "AutoencoderDetector":
        """
        Train the autoencoder on (mostly) normal data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data. Can be a NumPy array or a pandas DataFrame.
        """
        self.model.train()

        X_np = self._to_numpy(X_train)
        X_tensor = torch.from_numpy(X_np)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            n_samples = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                recon = self.model(batch)
                loss_matrix = self.criterion(recon, batch)  # (batch, features)
                loss = loss_matrix.mean()                   # scalar

                loss.backward()
                self.optimizer.step()

                batch_size_eff = batch.size(0)
                epoch_loss += loss.item() * batch_size_eff
                n_samples += batch_size_eff

            _avg_loss = epoch_loss / max(n_samples, 1)
            # Optional: plug logger here if you want epoch-wise tracking.

        return self

    @torch.no_grad()
    def predict_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute reconstruction-error-based anomaly scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score. Can be a NumPy array or a pandas DataFrame.

        Returns
        -------
        np.ndarray
            Vector of anomaly scores. Higher values indicate more anomalous samples.
        """
        self.model.eval()

        X_np = self._to_numpy(X)
        X_tensor = torch.from_numpy(X_np)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_scores: list[np.ndarray] = []
        for (batch,) in loader:
            batch = batch.to(self.device)
            recon = self.model(batch)
            loss_matrix = self.criterion(recon, batch)  # (batch, features)
            scores = loss_matrix.mean(dim=1)
            all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)


# =============================================================================
# Supervised baselines: Decision Tree & Random Forest
# =============================================================================


class DecisionTreeDetector:
    """
    Supervised Decision Tree baseline.

    Interface:
        - fit(X, y)
        - predict_scores(X) -> probability of positive class (y=1)
    """

    def __init__(
        self,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 50,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ) -> None:
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
        )

    def fit(self, X, y) -> "DecisionTreeDetector":
        """Train supervised decision tree."""
        self.model.fit(X, y)
        return self

    def predict_scores(self, X) -> np.ndarray:
        """
        Return risk scores as P(y=1 | x).

        Handles the rare case where only one class is present in training.
        """
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]


class RandomForestDetector:
    """
    Supervised Random Forest baseline.

    Interface:
        - fit(X, y)
        - predict_scores(X) -> probability of positive class (y=1)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 50,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced_subsample",
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )

    def fit(self, X, y) -> "RandomForestDetector":
        """Train supervised random forest."""
        self.model.fit(X, y)
        return self

    def predict_scores(self, X) -> np.ndarray:
        """
        Return risk scores as P(y=1 | x).
        """
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]


# =============================================================================
# Spec-compliant helper aliases (kept for backwards compatibility)
# =============================================================================


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    """
    Specification-compliant alias for train_isolation_forest().
    """
    return train_isolation_forest(
        X_train,
        contamination=contamination,
        random_state=random_state,
    )


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Specification-compliant alias for get_if_anomaly_scores().
    """
    return get_if_anomaly_scores(model, X)
