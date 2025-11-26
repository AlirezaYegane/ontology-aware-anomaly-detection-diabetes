# Autoencoder Anomaly Detection - Jupyter Notebook Guide

This guide shows how to use the `autoencoder_anomaly_detection.py` module in a Jupyter notebook or Google Colab.

---

## 📋 Prerequisites

Make sure you have the following installed:
```bash
pip install numpy pandas torch scikit-learn matplotlib
```

---

## 🚀 Quick Start - Complete Pipeline

### Cell 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
from autoencoder_anomaly_detection import run_autoencoder_anomaly_detection_pipeline

# Assuming you already have X and y from preprocessing
# X should be a numpy array or pandas DataFrame
# y should be a binary array (1 = early readmission, 0 = normal)
```

### Cell 2: Load Your Preprocessed Data

```python
# Example: Load from preprocessing script
# If you've already run your preprocessing, X and y should be available

# Otherwise, load from saved files:
# X = np.load('X_preprocessed.npy')
# y = np.load('y_labels.npy')

# Or from pandas:
# df = pd.read_csv('preprocessed_data.csv')
# X = df.drop('readmitted', axis=1).values
# y = df['readmitted'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Anomaly rate: {y.mean():.2%}")
```

### Cell 3: Run Complete Pipeline

```python
# Run the complete autoencoder anomaly detection pipeline
results = run_autoencoder_anomaly_detection_pipeline(
    X=X,
    y=y,
    test_size=0.2,           # 20% test set
    bottleneck_dim=32,       # Latent dimension
    epochs=20,               # Training epochs
    compare_with_isolation_forest=True  # Compare with IsolationForest baseline
)
```

This single function call will:
- ✅ Perform stratified train/test split
- ✅ Extract normal samples (y=0)
- ✅ Build and train the autoencoder
- ✅ Compute reconstruction errors
- ✅ Evaluate with ROC-AUC and PR-AUC
- ✅ Generate comparison plots
- ✅ Compare with IsolationForest (optional)

### Cell 4: Access Results

```python
# Extract components from results
model = results['model']
ae_metrics = results['autoencoder_metrics']
test_errors = results['test_reconstruction_errors']
y_test = results['y_test']

print(f"Autoencoder ROC-AUC: {ae_metrics['roc_auc']:.4f}")
print(f"Autoencoder PR-AUC: {ae_metrics['pr_auc']:.4f}")

if results['isolation_forest_metrics'] is not None:
    if_metrics = results['isolation_forest_metrics']
    print(f"\nIsolationForest ROC-AUC: {if_metrics['roc_auc']:.4f}")
    print(f"IsolationForest PR-AUC: {if_metrics['pr_auc']:.4f}")
```

---

## 🔧 Advanced Usage - Step-by-Step

If you want more control over each step, you can use the individual functions:

### Cell 1: Import Individual Functions

```python
from autoencoder_anomaly_detection import (
    FeedforwardAutoencoder,
    train_autoencoder,
    compute_reconstruction_error,
    evaluate_anomaly_detector,
    plot_evaluation_curves,
    print_comparison_table
)
from sklearn.model_selection import train_test_split
import numpy as np
import torch
```

### Cell 2: Train/Test Split

```python
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train anomaly rate: {y_train.mean():.2%}")
print(f"Test anomaly rate: {y_test.mean():.2%}")
```

### Cell 3: Extract Normal Samples

```python
# Use only normal samples (y=0) for training
X_train_normal = X_train[y_train == 0]

print(f"Normal samples for training: {X_train_normal.shape[0]}")
print(f"Anomaly samples (excluded): {(y_train == 1).sum()}")
```

### Cell 4: Build and Train Autoencoder

```python
# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Train autoencoder
model, train_losses = train_autoencoder(
    X_train_normal=X_train_normal,
    input_dim=X_train.shape[1],
    bottleneck_dim=32,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    device=device,
    verbose=True
)
```

### Cell 5: Plot Training Loss

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, linewidth=2, color='#2E86AB')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
plt.title('Autoencoder Training Loss', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

### Cell 6: Compute Reconstruction Errors

```python
# Compute reconstruction errors as anomaly scores
train_errors = compute_reconstruction_error(model, X_train, device=device)
test_errors = compute_reconstruction_error(model, X_test, device=device)

print("Reconstruction Error Statistics:")
print(f"  Train mean: {train_errors.mean():.6f}")
print(f"  Test mean: {test_errors.mean():.6f}")
print(f"\nBy Class (Train):")
print(f"  Normal (y=0): {train_errors[y_train==0].mean():.6f}")
print(f"  Anomaly (y=1): {train_errors[y_train==1].mean():.6f}")
```

### Cell 7: Visualize Reconstruction Error Distribution

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Test set distribution
plt.subplot(1, 2, 1)
plt.hist(test_errors[y_test==0], bins=50, alpha=0.6, label='Normal (y=0)', color='#2E86AB')
plt.hist(test_errors[y_test==1], bins=50, alpha=0.6, label='Anomaly (y=1)', color='#A23B72')
plt.xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Test Set - Reconstruction Error Distribution', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot([test_errors[y_test==0], test_errors[y_test==1]], 
            labels=['Normal (y=0)', 'Anomaly (y=1)'],
            patch_artist=True,
            boxprops=dict(facecolor='#2E86AB', alpha=0.6))
plt.ylabel('Reconstruction Error', fontsize=12, fontweight='bold')
plt.title('Test Set - Error by Class', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Cell 8: Evaluate Performance

```python
# Evaluate autoencoder
ae_metrics = evaluate_anomaly_detector(
    y_true=y_test,
    anomaly_scores=test_errors,
    model_name="Autoencoder"
)
```

### Cell 9: Compare with IsolationForest

```python
from sklearn.ensemble import IsolationForest

# Train IsolationForest on normal samples
iso_forest = IsolationForest(
    contamination=y_train.mean(),
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal)

# Compute anomaly scores (negate to make higher = more anomalous)
if_test_scores = -iso_forest.score_samples(X_test)

# Evaluate
if_metrics = evaluate_anomaly_detector(
    y_true=y_test,
    anomaly_scores=if_test_scores,
    model_name="IsolationForest"
)

# Print comparison
print_comparison_table([if_metrics, ae_metrics])
```

### Cell 10: Plot Evaluation Curves

```python
# Plot ROC and Precision-Recall curves
plot_evaluation_curves(y_test, test_errors, if_test_scores)
```

---

## 🎯 Using Custom Anomaly Thresholds

### Example: Flag Top 5% as Anomalies

```python
# Define threshold at 95th percentile
threshold = np.percentile(test_errors, 95)

# Predict anomalies
y_pred = (test_errors >= threshold).astype(int)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 💾 Save and Load Model

### Save Model

```python
# Save the trained autoencoder
torch.save(model.state_dict(), 'autoencoder_model.pth')
print("✓ Model saved!")
```

### Load Model

```python
# Load the model
input_dim = X.shape[1]
bottleneck_dim = 32

loaded_model = FeedforwardAutoencoder(input_dim, bottleneck_dim)
loaded_model.load_state_dict(torch.load('autoencoder_model.pth'))
loaded_model.eval()

print("✓ Model loaded!")
```

---

## 📊 Expected Output

When you run the complete pipeline, you should see:

1. **Console Output:**
   - Train/test split info
   - Training progress (loss per epoch)
   - Reconstruction error statistics
   - Evaluation metrics (ROC-AUC, PR-AUC)
   - Performance at different thresholds
   - Comparison table

2. **Visualizations:**
   - **ROC Curve:** Shows true positive rate vs false positive rate
   - **Precision-Recall Curve:** Shows precision vs recall trade-off
   - Both curves compare Autoencoder vs IsolationForest

3. **Typical Results:**
   - ROC-AUC: 0.60 - 0.75 (depending on data)
   - PR-AUC: 0.15 - 0.35 (higher than baseline)
   - Autoencoder often performs slightly better than IsolationForest on this task

---

## ⚙️ Hyperparameter Tuning

You can experiment with different hyperparameters:

```python
# Try different configurations
configs = [
    {'bottleneck_dim': 16, 'epochs': 15},
    {'bottleneck_dim': 32, 'epochs': 20},
    {'bottleneck_dim': 64, 'epochs': 25},
]

for config in configs:
    print(f"\n{'='*60}")
    print(f"Testing config: {config}")
    print(f"{'='*60}")
    
    results = run_autoencoder_anomaly_detection_pipeline(
        X=X, y=y,
        bottleneck_dim=config['bottleneck_dim'],
        epochs=config['epochs'],
        compare_with_isolation_forest=False  # Skip for speed
    )
    
    print(f"ROC-AUC: {results['autoencoder_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {results['autoencoder_metrics']['pr_auc']:.4f}")
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Use CPU instead:
```python
device = 'cpu'  # Force CPU usage
```

### Issue: Poor performance (ROC-AUC < 0.55)
**Solutions:**
1. Try different bottleneck dimensions (16, 32, 64)
2. Increase epochs (30-50)
3. Check data scaling (autoencoder works best with scaled data)
4. Verify that normal class (y=0) is actually the majority

### Issue: NaN losses during training
**Solutions:**
1. Reduce learning rate: `learning_rate=0.0001`
2. Check for NaN/Inf values in data
3. Ensure data is properly scaled

---

## 📝 Notes

- The autoencoder is trained **only** on normal samples (y=0)
- Higher reconstruction error indicates more anomalous samples
- The model is CPU-friendly and should run in <5 minutes on most machines
- For production use, consider using a validation set for early stopping

---

## 🎓 Understanding the Results

**ROC-AUC** (Receiver Operating Characteristic - Area Under Curve):
- Measures overall discriminative ability
- Range: 0.5 (random) to 1.0 (perfect)
- >0.7 is generally considered good for this problem

**PR-AUC** (Precision-Recall - Area Under Curve):
- More informative for imbalanced datasets
- Range: baseline (proportion of anomalies) to 1.0
- Higher values mean better precision-recall trade-off

**Percentile thresholds:**
- 90th: Flags top 10% as anomalies (high recall, lower precision)
- 95th: Flags top 5% as anomalies (balanced)
- 99th: Flags top 1% as anomalies (high precision, lower recall)

Choose threshold based on your use case:
- **Hospital setting:** High recall (90th) to catch most readmissions
- **Resource-constrained:** High precision (99th) to focus on most likely cases

---

**Happy Anomaly Detecting! 🚀**
