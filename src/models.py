"""
Models module for Diabetes Hospital Readmission Anomaly Detection.
Includes Autoencoder and IsolationForest implementations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from typing import Tuple, List, Optional

# ============================================================================
# Autoencoder
# ============================================================================

class FeedforwardAutoencoder(nn.Module):
    """
    Simple feedforward autoencoder for anomaly detection.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int = 32):
        super(FeedforwardAutoencoder, self).__init__()
        
        # Calculate hidden dimension
        hidden_dim = max(64, input_dim // 2)
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder layers (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X_train_normal: np.ndarray, input_dim: int, bottleneck_dim: int = 32, 
                     epochs: int = 20, batch_size: int = 256, learning_rate: float = 0.001, 
                     device: str = 'cpu', verbose: bool = True) -> Tuple[nn.Module, List[float]]:
    """
    Train the autoencoder on normal samples only.
    """
    # Initialize model
    model = FeedforwardAutoencoder(input_dim, bottleneck_dim).to(device)
    
    # Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    X_tensor = torch.FloatTensor(X_train_normal).to(device)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    train_losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y in dataloader:
            # Forward pass
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model, train_losses

def compute_reconstruction_error(model: nn.Module, X: np.ndarray, device: str = 'cpu', batch_size: int = 1024) -> np.ndarray:
    """
    Compute reconstruction error for all samples in X.
    """
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_X = batch[0]
            reconstructed = model(batch_X)
            
            # Compute MSE for each sample
            mse = ((batch_X - reconstructed) ** 2).mean(dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())
    
    return np.array(reconstruction_errors)

# ============================================================================
# Isolation Forest
# ============================================================================

def train_isolation_forest(X_train: np.ndarray, contamination: float = 0.1, random_state: int = 42) -> IsolationForest:
    """
    Train Isolation Forest model.
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    return iso_forest

def get_if_anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Get anomaly scores from Isolation Forest (negated so higher = more anomalous).
    """
    return -model.score_samples(X)
