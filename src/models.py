"""
Models module for Diabetes Hospital Readmission Anomaly Detection.
Includes:
- IsolationForestDetector  (wrapper around sklearn IsolationForest)
- AutoencoderDetector      (PyTorch feed-forward autoencoder)
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.ensemble import IsolationForest

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

    encoder:  input_dim -> h1 -> h2 -> ... -> bottleneck
    decoder:  bottleneck -> ... -> h2 -> h1 -> input_dim
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
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
# Isolation Forest Detector
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
    Get anomaly scores from Isolation Forest.

    sklearn's IsolationForest.score_samples:
        - higher score = more "normal"
    در اینجا برای anomaly score:
        - بالاتر = مشکوک‌تر
    پس منفی‌اش می‌کنیم.
    """
    scores = -model.score_samples(X)
    return scores


class IsolationForestDetector:
    """
    Wrapper used by run_pipeline_direct.py

    Methods:
        - fit(X_train)
        - predict_scores(X)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        n_estimators: int = 100,
    ):
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
# Autoencoder Detector
# =============================================================================

class AutoencoderDetector:
    """
    High-level detector API for the autoencoder model.

    Used by run_pipeline_direct.py as:

        ae = AutoencoderDetector(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32],
            epochs=50,
            batch_size=256,
            learning_rate=1e-3,
        )
        ae.fit(X_train)
        scores = ae.predict_scores(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
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

    def fit(self, X_train) -> "AutoencoderDetector":
        """
        Train autoencoder on (mostly) normal data.

        X_train: numpy array OR pandas DataFrame, shape (n_samples, n_features)
        """
        self.model.train()

        # 🔧 قبول کردن هم numpy هم DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.to_numpy(dtype=np.float32)
        else:
            X_np = X_train.astype(np.float32)

        X = torch.from_numpy(X_np)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_samples = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                recon = self.model(batch)
                loss_matrix = self.criterion(recon, batch)  # shape (batch, features)
                loss = loss_matrix.mean()                   # scalar

                loss.backward()
                self.optimizer.step()

                batch_size_effective = batch.size(0)
                epoch_loss += loss.item() * batch_size_effective
                n_samples += batch_size_effective

            avg_loss = epoch_loss / max(n_samples, 1)
            # اگر خواستی، این پرینت دیباگ را فعال کن:
            # print(f"[Autoencoder] Epoch {epoch+1}/{self.epochs} - loss={avg_loss:.6f}")

        return self


    @torch.no_grad()
    def predict_scores(self, X) -> np.ndarray:
        """
        Compute reconstruction-error-based anomaly scores.

        X: numpy array OR pandas DataFrame
        Output:
            scores[i] = MSE reconstruction error for sample i
            بالاتر = مشکوک‌تر
        """
        self.model.eval()

        # 🔧 قبول کردن DataFrame و numpy
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy(dtype=np.float32)
        else:
            X_np = X.astype(np.float32)

        X_tensor = torch.from_numpy(X_np)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_scores = []
        for (batch,) in loader:
            batch = batch.to(self.device)
            recon = self.model(batch)
            loss_matrix = self.criterion(recon, batch)  # (batch, features)
            scores = loss_matrix.mean(dim=1)
            all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0)


# =============================================================================
# "Specification-compliant" aliases (in case other code uses them)
# =============================================================================

def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest model.
    Alias for train_isolation_forest() - specification-compliant name.
    """
    return train_isolation_forest(X_train, contamination, random_state)


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Get anomaly scores from Isolation Forest.
    Alias for get_if_anomaly_scores() - specification-compliant name.
    """
    return get_if_anomaly_scores(model, X)
