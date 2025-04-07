import os
import sys
import torch
import torch.nn as nn
import numpy as np
# import argparse # Removed argparse import
from pathlib import Path
import matplotlib.pyplot as plt

import sys # Keep sys for potential exit()
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_DATA_DIR, TEST_FRAMES, VALIDATION_FRAMES, # Removed REAL_SENSE_SEQUENCES
    POINT_CLOUD_PARAMS, MODEL_PARAMS, WEIGHTS_DIR, RESULTS_DIR, # Added MODEL_PARAMS
    UCL_DATA_CONFIG, # Import the UCL dataset config
    # Import evaluation parameters (previously CLI args or defaults)
    EVAL_CHECKPOINT, EVAL_TEST_SET, EVAL_MODEL_TYPE, EVAL_K,
    EVAL_BATCH_SIZE, EVAL_VISUALIZE, EVAL_NUM_VISUALIZATIONS,
    EVAL_RESULTS_DIR, # Optional results dir from config
    # General config params
    SEED, DEVICE, NUM_WORKERS
)
from models.classifier import get_model
from models.utils import (
    load_checkpoint, compute_metrics, plot_confusion_matrix, set_seed
)
from data_processing.dataset import TableDataset
from data_processing.depth_to_pointcloud import create_rgbd_pointcloud, visualize_pointcloud, create_pointcloud_from_depth
from data_processing.preprocessing import preprocess_point_cloud

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (loss, metrics, predictions) where predictions is a dictionary of 
            file paths and their predictions
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Get inputs and labels
            points = data['points'].to(device)
            labels = data['label'].to(device)
            metadata = data['metadata']  # This is a list of dictionaries
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Track loss
            running_loss += loss.item()
            
            # Track predictions
            preds = torch.argmax(outputs, dim=1)
            scores = torch.softmax(outputs, dim=1)[:, 1]
            
            # Extend lists for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.detach().cpu().numpy())
            
            # Store predictions for each sample
            for i in range(len(preds)):
                if i < len(metadata):  # Ensure we don't go out of bounds
                    depth_file = metadata[i]['depth_file']
                    predictions[depth_file] = {
                        'pred': preds[i].item(),
                        'score': scores[i].item(),
                        'label': labels[i].item(),
                        'sequence': metadata[i]['sequence'],
                        'sub_sequence': metadata[i]['sub_sequence'],
                        'image_file': metadata[i].get('image_file')
                    }
    
    # Calculate metrics
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores))
    
    # Average loss
    avg_loss = running_loss / len(test_loader)
    
    return avg_loss, metrics, predictions

def visualize_predictions(predictions, results_dir, num_samples=5):
    """Visualize some predictions.
    
    Args:
        predictions (dict): Dictionary of predictions
        results_dir (str): Directory to save results
        num_samples (int): Number of samples to visualize
    """
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Get some correct and incorrect predictions
    correct_preds = [k for k, v in predictions.items() if v['pred'] == v['label']]
    incorrect_preds = [k for k, v in predictions.items() if v['pred'] != v['label']]
    
    # Randomly sample from each
    n_correct = min(num_samples, len(correct_preds))
    n_incorrect = min(num_samples, len(incorrect_preds))
    
    if n_correct > 0:
        sampled_correct = np.random.choice(correct_preds, n_correct, replace=False)
        
        for i, depth_file in enumerate(sampled_correct):
            pred_data = predictions[depth_file]
            
            # Get paths
            intrinsics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(depth_file))), 'intrinsics.txt')
            image_file = pred_data.get('image_file')
            
            # Create title
            title = f"Correct: Pred={pred_data['pred']} (Score={pred_data['score']:.2f}), Label={pred_data['label']}"
            
            # Visualize point cloud
            try:
                if image_file and os.path.exists(image_file):
                    points, colors = create_rgbd_pointcloud(
                        depth_file, image_file, intrinsics_path, 
                        use_raw_depth=('harvard_tea_2' in depth_file)
                    )
                    
                    # Visualize colored point cloud
                    visualize_pointcloud(points, colors)
                else:
                    points = create_pointcloud_from_depth(
                        depth_file, intrinsics_path, 
                        use_raw_depth=('harvard_tea_2' in depth_file)
                    )
                    
                    # Visualize point cloud
                    visualize_pointcloud(points)
                
                print(title)
            except Exception as e:
                print(f"Error visualizing {depth_file}: {e}")
    
    if n_incorrect > 0:
        sampled_incorrect = np.random.choice(incorrect_preds, n_incorrect, replace=False)
        
        for i, depth_file in enumerate(sampled_incorrect):
            pred_data = predictions[depth_file]
            
            # Get paths
            intrinsics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(depth_file))), 'intrinsics.txt')
            image_file = pred_data.get('image_file')
            
            # Create title
            title = f"Incorrect: Pred={pred_data['pred']} (Score={pred_data['score']:.2f}), Label={pred_data['label']}"
            
            # Visualize point cloud
            try:
                if image_file and os.path.exists(image_file):
                    points, colors = create_rgbd_pointcloud(
                        depth_file, image_file, intrinsics_path, 
                        use_raw_depth=('harvard_tea_2' in depth_file)
                    )
                    
                    # Visualize colored point cloud
                    visualize_pointcloud(points, colors)
                else:
                    points = create_pointcloud_from_depth(
                        depth_file, intrinsics_path, 
                        use_raw_depth=('harvard_tea_2' in depth_file)
                    )
                    
                    # Visualize point cloud
                    visualize_pointcloud(points)
                
                print(title)
            except Exception as e:
                print(f"Error visualizing {depth_file}: {e}")

def analyze_results_by_sequence(predictions):
    """Analyze results by sequence.
    
    Args:
        predictions (dict): Dictionary of predictions
        
    Returns:
        dict: Dictionary of metrics by sequence
    """
    # Group predictions by sequence
    sequence_predictions = {}
    
    for _, pred_data in predictions.items():
        sequence = pred_data['sequence']
        if sequence not in sequence_predictions:
            sequence_predictions[sequence] = {
                'preds': [],
                'labels': [],
                'scores': []
            }
        
        sequence_predictions[sequence]['preds'].append(pred_data['pred'])
        sequence_predictions[sequence]['labels'].append(pred_data['label'])
        sequence_predictions[sequence]['scores'].append(pred_data['score'])
    
    # Compute metrics for each sequence
    sequence_metrics = {}
    
    for sequence, data in sequence_predictions.items():
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])
        scores = np.array(data['scores'])
        
        metrics = compute_metrics(labels, preds, scores)
        sequence_metrics[sequence] = metrics
    
    return sequence_metrics

def main():
    """Main function for evaluation. Reads all configuration from config.py."""
    # Set random seed for reproducibility
    set_seed(SEED) # Use SEED from config

    # Set device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu") # Use DEVICE from config
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not EVAL_CHECKPOINT or not os.path.exists(EVAL_CHECKPOINT):
        print(f"Error: Evaluation checkpoint path not found or not specified in config: {EVAL_CHECKPOINT}")
        sys.exit(1)

    # Create model using parameters from config
    model = get_model(
        model_type=EVAL_MODEL_TYPE, # Use config value
        num_classes=2,
        k=EVAL_K,                   # Use config value
        emb_dims=MODEL_PARAMS['emb_dims'],
        dropout=MODEL_PARAMS['dropout'],
        feature_dropout=MODEL_PARAMS.get('feature_dropout', 0.0)
    )

    # Move model to device
    model = model.to(device)

    # Load checkpoint
    model, _, _, _ = load_checkpoint(model, None, EVAL_CHECKPOINT) # Use config value

    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create test dataset loader
    from torch.utils.data import DataLoader
    from data_processing.dataset import collate_fn # Assuming collate_fn is defined here or imported

    dataset_name = f"Test Set {EVAL_TEST_SET}" # Use config value

    if EVAL_TEST_SET == 1: # Use config value
        # Use TEST_FRAMES list for Harvard test set
        if not TEST_FRAMES:
             print("Warning: TEST_FRAMES list is empty in config. Cannot evaluate Test Set 1.")
             return
        print(f"Evaluating on Test Set 1 (Harvard subset) using {len(TEST_FRAMES)} frame IDs.")
        test_dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            data_spec=TEST_FRAMES, # Pass the list of frame IDs
            augment=False,
            mode='test',
            point_cloud_params=POINT_CLOUD_PARAMS
        )
        dataset_name = "Test Set 1 (Harvard)"
    elif EVAL_TEST_SET == 2: # Use config value
        # Use UCL_DATA_CONFIG dictionary for the custom UCL test set (now Test Set 2)
        if not UCL_DATA_CONFIG:
             print("Warning: UCL_DATA_CONFIG is not defined or empty in config. Cannot evaluate Test Set 2 (UCL).")
             return
        print(f"Evaluating on Test Set 2 (UCL) using config: {UCL_DATA_CONFIG.get('name', 'UCL')}") # Use get for safety
        test_dataset = TableDataset(
            data_root=BASE_DATA_DIR, # data_root is still needed for relative path calculations inside dataset
            data_spec=UCL_DATA_CONFIG, # Pass the specific config dict
            augment=False,
            mode='test',
            point_cloud_params=POINT_CLOUD_PARAMS
        )
        dataset_name = f"Test Set 2 ({UCL_DATA_CONFIG.get('name', 'UCL')})"
    else:
        raise ValueError(f"Invalid test set specified in config (EVAL_TEST_SET): {EVAL_TEST_SET}. Choose 1 or 2.") # Use config value

    if len(test_dataset) == 0:
        print(f"Error: Test dataset for {dataset_name} is empty. Cannot evaluate.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE, # Use config value
        shuffle=False,
        num_workers=NUM_WORKERS,    # Use config value
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate model
    loss, metrics, predictions = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Print metrics
    print(f"Test Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    if metrics['auc_roc'] is not None:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

    # Plot confusion matrix
    results_dir = EVAL_RESULTS_DIR or RESULTS_DIR # Use config value or default
    os.makedirs(results_dir, exist_ok=True)

    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['No Table', 'Table'],
        title=f'Confusion Matrix - {dataset_name}',
        path=os.path.join(results_dir, f"confusion_matrix_test{EVAL_TEST_SET}.png") # Use config value
    )

    # Analyze results by sequence
    sequence_metrics = analyze_results_by_sequence(predictions)
    
    print("\nResults by Sequence:")
    for sequence, metrics in sequence_metrics.items():
        print(f"\n{sequence}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")

    # Visualize some predictions
    if EVAL_VISUALIZE: # Use config value
        print("\nVisualizing predictions...")
        visualize_predictions(
            predictions=predictions,
            results_dir=results_dir,
            num_samples=EVAL_NUM_VISUALIZATIONS # Use config value
        )

    print("Evaluation completed.")

if __name__ == "__main__":
    # Removed argparse setup and conditional logic
    main() # Call main directly
