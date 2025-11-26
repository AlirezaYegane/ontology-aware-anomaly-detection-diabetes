"""
Simple Example: Using Autoencoder for Anomaly Detection
========================================================

This script demonstrates how to use the autoencoder module
with your preprocessed diabetes dataset.

Run this after you've completed your preprocessing step.
"""

import numpy as np
import pandas as pd
from autoencoder_anomaly_detection import run_autoencoder_anomaly_detection_pipeline

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================
print("Loading preprocessed data...")

# If you've run the preprocessing and have X and y in memory, skip this step
# Otherwise, load them from your saved files or preprocessing script

# Example 1: If you saved as numpy arrays
try:
    X = np.load('X_preprocessed.npy')
    y = np.load('y_labels.npy')
    print("✓ Loaded data from .npy files")
except FileNotFoundError:
    print("⚠ .npy files not found. Trying alternative methods...")
    
    # Example 2: If you have a combined CSV
    try:
        df = pd.read_csv('preprocessed_diabetes_data.csv')
        
        # Assuming 'readmitted' is the target column
        if 'readmitted' in df.columns:
            y = df['readmitted'].values
            X = df.drop('readmitted', axis=1).values
            print("✓ Loaded data from CSV")
        else:
            raise ValueError("'readmitted' column not found in CSV")
    
    except FileNotFoundError:
        print("⚠ CSV file not found either.")
        print("\nPlease ensure you have preprocessed data available.")
        print("You can create dummy data for testing with:")
        print("  X = np.random.randn(10000, 50)")
        print("  y = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])")
        
        # Create dummy data for demonstration
        print("\nCreating dummy data for demonstration...")
        np.random.seed(42)
        X = np.random.randn(10000, 50)  # 10k samples, 50 features
        y = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])  # 10% anomalies
        print("✓ Dummy data created")

print(f"\nData Summary:")
print(f"  Features: {X.shape[1]}")
print(f"  Samples: {X.shape[0]}")
print(f"  Anomaly rate: {y.mean():.2%}")
print(f"  Normal samples: {(y==0).sum()}")
print(f"  Anomaly samples: {(y==1).sum()}")


# ============================================================================
# STEP 2: Run Complete Autoencoder Pipeline
# ============================================================================
print("\n" + "="*80)
print("Starting Autoencoder Anomaly Detection Pipeline")
print("="*80 + "\n")

results = run_autoencoder_anomaly_detection_pipeline(
    X=X,
    y=y,
    test_size=0.2,              # 20% for testing
    bottleneck_dim=32,          # Latent space dimension
    epochs=20,                  # Training epochs (increase for better results)
    compare_with_isolation_forest=True  # Compare with baseline
)


# ============================================================================
# STEP 3: Save Results (Optional)
# ============================================================================
print("\nSaving results...")

# Save the trained model
import torch
torch.save(results['model'].state_dict(), 'autoencoder_model.pth')
print("✓ Model saved to: autoencoder_model.pth")

# Save anomaly scores
np.save('test_reconstruction_errors.npy', results['test_reconstruction_errors'])
print("✓ Test reconstruction errors saved")

# Save metrics to a text file
with open('autoencoder_metrics.txt', 'w') as f:
    f.write("Autoencoder Anomaly Detection - Evaluation Metrics\n")
    f.write("="*60 + "\n\n")
    
    ae_metrics = results['autoencoder_metrics']
    f.write(f"Autoencoder:\n")
    f.write(f"  ROC-AUC: {ae_metrics['roc_auc']:.4f}\n")
    f.write(f"  PR-AUC:  {ae_metrics['pr_auc']:.4f}\n\n")
    
    if results['isolation_forest_metrics'] is not None:
        if_metrics = results['isolation_forest_metrics']
        f.write(f"IsolationForest:\n")
        f.write(f"  ROC-AUC: {if_metrics['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:  {if_metrics['pr_auc']:.4f}\n\n")
        
        # Calculate improvement
        improvement = ae_metrics['roc_auc'] - if_metrics['roc_auc']
        f.write(f"Improvement (AE vs IF): {improvement:+.4f}\n")

print("✓ Metrics saved to: autoencoder_metrics.txt")


# ============================================================================
# STEP 4: Additional Analysis (Optional)
# ============================================================================
print("\n" + "="*80)
print("Additional Analysis")
print("="*80)

# Analyze reconstruction errors by class
test_errors = results['test_reconstruction_errors']
y_test = results['y_test']

normal_errors = test_errors[y_test == 0]
anomaly_errors = test_errors[y_test == 1]

print(f"\nReconstruction Error Analysis:")
print(f"  Normal samples (y=0):")
print(f"    Mean:   {normal_errors.mean():.6f}")
print(f"    Median: {np.median(normal_errors):.6f}")
print(f"    Std:    {normal_errors.std():.6f}")
print(f"\n  Anomaly samples (y=1):")
print(f"    Mean:   {anomaly_errors.mean():.6f}")
print(f"    Median: {np.median(anomaly_errors):.6f}")
print(f"    Std:    {anomaly_errors.std():.6f}")

# Calculate separation
separation = (anomaly_errors.mean() - normal_errors.mean()) / normal_errors.std()
print(f"\n  Separation (effect size): {separation:.2f}")
print(f"  (Higher is better - how many std devs apart)")

# Find optimal threshold using Youden's J statistic
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, test_errors)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n  Optimal threshold (Youden): {optimal_threshold:.6f}")
print(f"  At this threshold:")
print(f"    TPR (Recall):    {tpr[optimal_idx]:.3f}")
print(f"    FPR:             {fpr[optimal_idx]:.3f}")
print(f"    J-score:         {j_scores[optimal_idx]:.3f}")

# Create predictions at optimal threshold
y_pred_optimal = (test_errors >= optimal_threshold).astype(int)

from sklearn.metrics import classification_report
print(f"\n  Classification Report at Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal, 
                          target_names=['Normal', 'Anomaly'],
                          digits=3))


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. autoencoder_model.pth - Trained model weights")
print("  2. test_reconstruction_errors.npy - Anomaly scores for test set")
print("  3. autoencoder_metrics.txt - Performance metrics")
print("\nYou can now:")
print("  • Use the model for inference on new data")
print("  • Experiment with different thresholds")
print("  • Try different hyperparameters (bottleneck_dim, epochs)")
print("  • Integrate into your production pipeline")
print("="*80)
