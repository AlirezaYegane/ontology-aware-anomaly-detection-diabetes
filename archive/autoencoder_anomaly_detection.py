"""
Autoencoder-based Anomaly Detection for Readmission Prediction
================================================================

This module implements a feedforward autoencoder as a baseline anomaly detector
for identifying early hospital readmissions (<30 days) in the diabetes dataset.

The autoencoder is trained ONLY on normal samples (y=0, not early readmission)
and uses reconstruction error as an anomaly score.

Dependencies:
- numpy
- pandas
- torch
- scikit-learn
- matplotlib

Author: Senior ML Engineer
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score
)
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# SECTION 1: Autoencoder Architecture
# ============================================================================

class FeedforwardAutoencoder(nn.Module):
    """
    Simple feedforward autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: input -> hidden_dim/2 -> bottleneck_dim
    - Decoder: bottleneck_dim -> hidden_dim/2 -> input
    - Activation: ReLU for hidden layers, Sigmoid for output (assuming scaled data)
    
    The model learns to reconstruct normal samples. Anomalies will have
    higher reconstruction error.
    """
    
    def __init__(self, input_dim, bottleneck_dim=32):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim (int): Number of input features
            bottleneck_dim (int): Dimension of the bottleneck layer (latent space)
        """
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
            # No activation here - we'll apply it based on data scaling
        )
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================================
# SECTION 2: Training and Reconstruction Functions
# ============================================================================

def train_autoencoder(X_train_normal, input_dim, bottleneck_dim=32, 
                     epochs=20, batch_size=256, learning_rate=0.001, 
                     device='cpu', verbose=True):
    """
    Train the autoencoder on normal samples only.
    
    Args:
        X_train_normal (np.ndarray): Training data with only normal samples (y=0)
        input_dim (int): Number of input features
        bottleneck_dim (int): Bottleneck dimension
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for Adam optimizer
        device (str): 'cpu' or 'cuda'
        verbose (bool): Whether to print training progress
    
    Returns:
        model: Trained autoencoder model
        train_losses: List of training losses per epoch
    """
    # Initialize model
    model = FeedforwardAutoencoder(input_dim, bottleneck_dim).to(device)
    
    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()
    
    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    X_tensor = torch.FloatTensor(X_train_normal).to(device)
    dataset = TensorDataset(X_tensor, X_tensor)  # Input = target for autoencoder
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
    
    if verbose:
        print(f"\n✓ Training complete! Final loss: {train_losses[-1]:.6f}")
    
    return model, train_losses


def compute_reconstruction_error(model, X, device='cpu', batch_size=1024):
    """
    Compute reconstruction error for all samples in X.
    
    Args:
        model: Trained autoencoder model
        X (np.ndarray): Input data
        device (str): 'cpu' or 'cuda'
        batch_size (int): Batch size for inference
    
    Returns:
        reconstruction_errors (np.ndarray): MSE reconstruction error for each sample
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
# SECTION 3: Evaluation Functions
# ============================================================================

def evaluate_anomaly_detector(y_true, anomaly_scores, model_name="Model"):
    """
    Evaluate anomaly detector using various metrics.
    
    Args:
        y_true (np.ndarray): True labels (1=anomaly, 0=normal)
        anomaly_scores (np.ndarray): Anomaly scores (higher = more anomalous)
        model_name (str): Name of the model for display
    
    Returns:
        metrics (dict): Dictionary of evaluation metrics
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


def plot_evaluation_curves(y_test, autoencoder_scores, isolation_forest_scores=None):
    """
    Plot ROC and Precision-Recall curves for anomaly detectors.
    
    Args:
        y_test (np.ndarray): True labels
        autoencoder_scores (np.ndarray): Autoencoder anomaly scores
        isolation_forest_scores (np.ndarray, optional): IsolationForest anomaly scores
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # === ROC Curve ===
    ax = axes[0]
    
    # Autoencoder ROC
    fpr_ae, tpr_ae, _ = roc_curve(y_test, autoencoder_scores)
    roc_auc_ae = roc_auc_score(y_test, autoencoder_scores)
    ax.plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC = {roc_auc_ae:.3f})', 
            linewidth=2, color='#2E86AB')
    
    # IsolationForest ROC (if provided)
    if isolation_forest_scores is not None:
        fpr_if, tpr_if, _ = roc_curve(y_test, isolation_forest_scores)
        roc_auc_if = roc_auc_score(y_test, isolation_forest_scores)
        ax.plot(fpr_if, tpr_if, label=f'IsolationForest (AUC = {roc_auc_if:.3f})', 
                linewidth=2, color='#A23B72', linestyle='--')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # === Precision-Recall Curve ===
    ax = axes[1]
    
    # Autoencoder PR
    precision_ae, recall_ae, _ = precision_recall_curve(y_test, autoencoder_scores)
    pr_auc_ae = average_precision_score(y_test, autoencoder_scores)
    ax.plot(recall_ae, precision_ae, label=f'Autoencoder (AUC = {pr_auc_ae:.3f})', 
            linewidth=2, color='#2E86AB')
    
    # IsolationForest PR (if provided)
    if isolation_forest_scores is not None:
        precision_if, recall_if, _ = precision_recall_curve(y_test, isolation_forest_scores)
        pr_auc_if = average_precision_score(y_test, isolation_forest_scores)
        ax.plot(recall_if, precision_if, label=f'IsolationForest (AUC = {pr_auc_if:.3f})', 
                linewidth=2, color='#A23B72', linestyle='--')
    
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


def print_comparison_table(metrics_list):
    """
    Print a comparison table of multiple models.
    
    Args:
        metrics_list (list): List of metric dictionaries from evaluate_anomaly_detector
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


# ============================================================================
# SECTION 4: Complete Pipeline Function
# ============================================================================

def run_autoencoder_anomaly_detection_pipeline(X, y, test_size=0.2, 
                                              bottleneck_dim=32, epochs=20,
                                              compare_with_isolation_forest=True):
    """
    Complete pipeline for autoencoder-based anomaly detection.
    
    This function:
    1. Splits data into train/test with stratification
    2. Trains autoencoder on normal samples only (y_train=0)
    3. Computes reconstruction errors as anomaly scores
    4. Evaluates performance using ROC-AUC and PR-AUC
    5. Plots evaluation curves
    6. Optionally compares with IsolationForest
    
    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix
        y (np.ndarray or pd.Series): Binary labels (1=anomaly/early readmission, 0=normal)
        test_size (float): Proportion of test set
        bottleneck_dim (int): Bottleneck dimension for autoencoder
        epochs (int): Number of training epochs
        compare_with_isolation_forest (bool): Whether to train and compare with IsolationForest
    
    Returns:
        results (dict): Dictionary containing model, scores, and metrics
    """
    print("="*80)
    print("AUTOENCODER-BASED ANOMALY DETECTION PIPELINE")
    print("="*80)
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # -------------------------------------------------------------------------
    # STEP 1: Stratified Train/Test Split
    # -------------------------------------------------------------------------
    print("\n[1/6] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"  → Train set: {X_train.shape[0]} samples")
    print(f"  → Test set:  {X_test.shape[0]} samples")
    print(f"  → Train anomaly rate: {y_train.mean():.2%}")
    print(f"  → Test anomaly rate:  {y_test.mean():.2%}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Extract Normal Samples for Training
    # -------------------------------------------------------------------------
    print("\n[2/6] Extracting normal samples (y=0) for autoencoder training...")
    X_train_normal = X_train[y_train == 0]
    print(f"  → Normal samples in train set: {X_train_normal.shape[0]}")
    print(f"  → Anomaly samples in train set: {(y_train == 1).sum()} (NOT used for training)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Train Autoencoder
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Training autoencoder on normal samples...")
    print(f"  → Input dimension: {X.shape[1]}")
    print(f"  → Bottleneck dimension: {bottleneck_dim}")
    print(f"  → Epochs: {epochs}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  → Device: {device}")
    
    model, train_losses = train_autoencoder(
        X_train_normal=X_train_normal,
        input_dim=X.shape[1],
        bottleneck_dim=bottleneck_dim,
        epochs=epochs,
        batch_size=256,
        learning_rate=0.001,
        device=device,
        verbose=True
    )
    
    # -------------------------------------------------------------------------
    # STEP 4: Compute Reconstruction Errors
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Computing reconstruction errors...")
    
    train_reconstruction_errors = compute_reconstruction_error(model, X_train, device=device)
    test_reconstruction_errors = compute_reconstruction_error(model, X_test, device=device)
    
    print(f"  → Train reconstruction error (mean): {train_reconstruction_errors.mean():.6f}")
    print(f"  → Test reconstruction error (mean):  {test_reconstruction_errors.mean():.6f}")
    print(f"  → Normal samples (train, y=0) error: {train_reconstruction_errors[y_train==0].mean():.6f}")
    print(f"  → Anomaly samples (train, y=1) error: {train_reconstruction_errors[y_train==1].mean():.6f}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Evaluate Autoencoder
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Evaluating autoencoder on test set...")
    
    ae_metrics = evaluate_anomaly_detector(
        y_true=y_test,
        anomaly_scores=test_reconstruction_errors,
        model_name="Autoencoder"
    )
    
    # -------------------------------------------------------------------------
    # STEP 6: Train and Evaluate IsolationForest (if requested)
    # -------------------------------------------------------------------------
    if_scores = None
    if_metrics = None
    
    if compare_with_isolation_forest:
        print(f"\n[6/6] Training IsolationForest for comparison...")
        
        # Train IsolationForest on normal samples (same as autoencoder)
        iso_forest = IsolationForest(
            contamination=y_train.mean(),  # Expected proportion of anomalies
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_normal)
        
        # Compute anomaly scores (negate to make higher = more anomalous)
        if_scores = -iso_forest.score_samples(X_test)
        
        if_metrics = evaluate_anomaly_detector(
            y_true=y_test,
            anomaly_scores=if_scores,
            model_name="IsolationForest"
        )
        
        # Print comparison table
        print_comparison_table([if_metrics, ae_metrics])
    
    # -------------------------------------------------------------------------
    # Plot Evaluation Curves
    # -------------------------------------------------------------------------
    print("Generating evaluation plots...")
    plot_evaluation_curves(y_test, test_reconstruction_errors, if_scores)
    
    # -------------------------------------------------------------------------
    # Return Results
    # -------------------------------------------------------------------------
    results = {
        'model': model,
        'train_losses': train_losses,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_reconstruction_errors': train_reconstruction_errors,
        'test_reconstruction_errors': test_reconstruction_errors,
        'autoencoder_metrics': ae_metrics,
        'isolation_forest_scores': if_scores,
        'isolation_forest_metrics': if_metrics
    }
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    
    return results


# ============================================================================
# EXAMPLE USAGE (for Jupyter Notebook)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage assuming you have preprocessed X and y available.
    
    In your Jupyter notebook, you would run:
    
    ```python
    # Assuming X and y are already loaded from preprocessing
    results = run_autoencoder_anomaly_detection_pipeline(
        X=X,
        y=y,
        test_size=0.2,
        bottleneck_dim=32,
        epochs=20,
        compare_with_isolation_forest=True
    )
    
    # Access results
    model = results['model']
    metrics = results['autoencoder_metrics']
    test_scores = results['test_reconstruction_errors']
    ```
    """
    print(__doc__)
