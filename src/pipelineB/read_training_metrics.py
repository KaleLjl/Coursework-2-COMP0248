"""
This script 
  - prints the best training metrics from the training log CSV file.

Requirements:
  - Change the path configurations if needed.
"""
# ========== Imports ================
import pandas as pd

# Path to training metrics CSV
csv_path = "./results/pipelineB/PRETRAINR18_15_32/train_metrics.csv"  # update this path if needed

# Load CSV
df = pd.read_csv(csv_path)

best_idx = df["val_acc"].idxmax()
best_row = df.loc[best_idx]

# Print best metrics
print("===== Best Model (Based on Best Validation Accuracy) =====")
print(f"Epoch       : {int(best_row['epoch'])}")
print(f"Train Loss  : {best_row['train_loss']:.4f}")
print(f"Val Loss    : {best_row['val_loss']:.4f}")
print(f"Train Acc   : {best_row['train_acc']:.4f}")
print(f"Val Acc     : {best_row['val_acc']:.4f}")
print(f"Train F1    : {best_row['train_f1']:.4f}")
print(f"Val F1      : {best_row['val_f1']:.4f}")
print(f"Train Precision : {best_row['train_precision']:.4f}")
print(f"Val Precision   : {best_row['val_precision']:.4f}")
print(f"Train Recall    : {best_row['train_recall']:.4f}")
print(f"Val Recall      : {best_row['val_recall']:.4f}")
