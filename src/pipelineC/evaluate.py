import os
import sys
import argparse

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm

from src.pipelineC.model import DGCNN_Seg
from src.pipelineC.dataloader import get_dataloaders
from src.pipelineC.utils import compute_metrics
from src.pipelineC.visualize import plot_metrics_comparison


def evaluate(model, dataloader, criterion, device, output_dir):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        output_dir (str): Directory to save outputs

    Returns:
        tuple: (test_loss, test_metrics)
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Progress bar for evaluation
    pbar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            # Get data
            inputs = data['point_cloud'].to(device)
            targets = data['labels'].to(device)

            inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Update loss
            test_loss += loss.item()

            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)

            # Update progress bar
            pbar.set_postfix({
                'test_loss': f"{loss.item():.4f}",
                'avg_loss': f"{test_loss / (batch_idx + 1):.4f}"
            })

            # Store predictions and targets for metrics calculation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute average loss
    test_loss /= len(dataloader)

    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])

    test_metrics = compute_metrics(flat_targets, flat_preds)

    # Log segmentation statistics
    table_points_gt = np.sum(flat_targets == 1)
    table_points_pred = np.sum(flat_preds == 1)
    bg_points_gt = np.sum(flat_targets == 0)
    bg_points_pred = np.sum(flat_preds == 0)
    total_points = len(flat_targets)

    # Print evaluation summary with standardized format
    print("\nTest Results:")
    print(f"  Loss:            {test_loss:.4f}")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {test_metrics['mean_iou']:.4f}")
    print(f"  Background IoU:  {test_metrics['iou_background']:.4f}")
    print(f"  Table IoU:       {test_metrics['iou_table']:.4f}")
    print(f"  F1 Score:        {test_metrics.get('f1_weighted', 0.0):.4f}")

    # Print point distribution statistics
    print("\nPoint Distribution:")
    print(f"  Ground Truth: Background: {bg_points_gt} ({bg_points_gt/total_points*100:.2f}%), "
          f"Table: {table_points_gt} ({table_points_gt/total_points*100:.2f}%)")
    print(f"  Predictions: Background: {bg_points_pred} ({bg_points_pred/total_points*100:.2f}%), "
          f"Table: {table_points_pred} ({table_points_pred/total_points*100:.2f}%)")

    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': test_loss,
            'accuracy': float(test_metrics['accuracy']),
            'mean_iou': float(test_metrics['mean_iou']),
            'iou_table': float(test_metrics['iou_table']),
            'iou_background': float(test_metrics['iou_background']),
            'f1_weighted': float(test_metrics.get('f1_weighted', 0.0)),
            'precision_weighted': float(test_metrics.get('precision_weighted', 0.0)),
            'recall_weighted': float(test_metrics.get('recall_weighted', 0.0)),
            'point_distribution': {
                'ground_truth': {
                    'background': int(bg_points_gt),
                    'table': int(table_points_gt),
                    'background_percent': float(bg_points_gt/total_points*100),
                    'table_percent': float(table_points_gt/total_points*100)
                },
                'predictions': {
                    'background': int(bg_points_pred),
                    'table': int(table_points_pred),
                    'background_percent': float(bg_points_pred/total_points*100),
                    'table_percent': float(table_points_pred/total_points*100)
                }
            }
        }, f, indent=4)

    print(f"Metrics saved to {metrics_file}")

    # Plot metrics
    metrics_plot_file = os.path.join(output_dir, 'metrics_plot.png')
    metrics_to_plot = {
        'Accuracy': test_metrics['accuracy'],
        'Mean IoU': test_metrics['mean_iou'],
        'Table IoU': test_metrics['iou_table'],
        'Background IoU': test_metrics['iou_background'],
        'F1 Score': test_metrics.get('f1_weighted', 0.0)
    }
    plot_metrics_comparison(
        metrics_dict=metrics_to_plot,
        title="Segmentation Performance Metrics",
        save_path=metrics_plot_file
    )

    return test_loss, test_metrics


def main(args):
    """
    Main function for evaluating Pipeline C.

    Args:
        args: Command line arguments
    """

    # Set up output files
    output_dir = os.path.join("results/pipelineC", 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataloaders
    _, _, test_loader = get_dataloaders(args.test_set)

    # Get model
    model = DGCNN_Seg()
    model = model.to(device)

    # Load checkpoint
    checkpoint_dir = "weights/pipelineC"
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")

    # Get loss weights
    class_weights = torch.tensor([1, 1000], dtype=torch.float32).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluate model
    _, _ = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
    )

    print("Evaluation completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Pipeline C: Point Cloud Segmentation')
    parser.add_argument('--test_set', type=int, default=1,
                        help='Test set to use (1: Harvard, 2: RealSense)')
    args = parser.parse_args()
    main(args)
