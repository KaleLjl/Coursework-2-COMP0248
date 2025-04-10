"""
This script 
  - loads a CSV file containing training and validation metrics,
  - visualizes the metrics over epochs

References:
  - Provided lab materials.

Requirements:
  - Change the path configurations if needed.
"""
# ========== Imports ==========================
import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_and_save_metrics(csv_path, output_dir):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Print best validation accuracy
    best_val_idx = df["val_acc"].idxmax()
    best_val_acc = df.loc[best_val_idx, "val_acc"]
    best_val_epoch = int(df.loc[best_val_idx, "epoch"])

    best_train_idx = df["train_acc"].idxmax()
    best_train_acc = df.loc[best_train_idx, "train_acc"]
    best_train_epoch = int(df.loc[best_train_idx, "epoch"])

    print(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}")
    print(f"Best Training Accuracy: {best_train_acc:.4f} at epoch {best_train_epoch}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics to plot
    metrics = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("F1 Score", "train_f1", "val_f1"),
        ("Precision", "train_precision", "val_precision"),
        ("Recall", "train_recall", "val_recall")
    ]

    for title, train_col, val_col in metrics:
        plt.figure()
        plt.plot(df["epoch"], df[train_col], label="Train", color="orange")
        plt.plot(df["epoch"], df[val_col], label="Validation", color="orangered")
        plt.title(f"Training and Validation {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        filename = f"{title.lower().replace(' ', '_')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Path to your CSV and where to save figures
    # csv_path = "./results/pipelineB/TRAIN/train_metrics.csv"
    # output_dir = "./results/pipelineB/TRAIN/metrics_figures"
    csv_path = "./results/pipelineB/PRETRAINR18_15_32/train_metrics.csv"
    output_dir = "./results/pipelineB/PRETRAINR18_15_32/metrics_figures"

    if os.path.exists(csv_path):
        visualize_and_save_metrics(csv_path, output_dir)
    else:
        print(f"CSV not found at: {csv_path}")
