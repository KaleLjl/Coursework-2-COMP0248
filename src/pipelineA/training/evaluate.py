import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_DATA_DIR, TEST1_SEQUENCES, TEST2_SEQUENCES,
    POINT_CLOUD_PARAMS, MODEL_PARAMS, WEIGHTS_DIR, RESULTS_DIR
)
from models.classifier import get_model
from models.utils import (
    load_checkpoint, compute_metrics, plot_confusion_matrix, set_seed
)
from data_processing.dataset import TableDataset
from data_processing.depth_to_pointcloud import create_rgbd_pointcloud, visualize_pointcloud
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
            for i in range(len(metadata)):
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

def main(args):
    """Main function for evaluation.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(
        model_type=args.model_type,
        num_classes=2,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=0.0  # No dropout for evaluation
    )
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint
    model, _, _, _ = load_checkpoint(model, None, args.checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create test dataset
    if args.test_set == 1:
        test_sequences = TEST1_SEQUENCES  # Harvard sequences
    elif args.test_set == 2:
        test_sequences = TEST2_SEQUENCES  # RealSense sequences
    else:
        raise ValueError(f"Invalid test set: {args.test_set}")
    
    from torch.utils.data import DataLoader
    from data_processing.dataset import collate_fn
    
    test_dataset = TableDataset(
        data_root=BASE_DATA_DIR,
        sequences=test_sequences,
        augment=False,
        mode='test',
        point_cloud_params=POINT_CLOUD_PARAMS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['No Table', 'Table'],
        title=f'Confusion Matrix - Test Set {args.test_set}',
        path=os.path.join(results_dir, f"confusion_matrix_test{args.test_set}.png")
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
    if args.visualize:
        print("\nVisualizing predictions...")
        visualize_predictions(
            predictions=predictions,
            results_dir=results_dir,
            num_samples=args.num_visualizations
        )
    
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate point cloud classifier")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="dgcnn",
                        choices=["dgcnn", "pointnet"],
                        help="Model type")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of nearest neighbors for DGCNN")
    parser.add_argument("--emb_dims", type=int, default=1024,
                        help="Embedding dimensions")
    
    # Evaluation parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_set", type=int, default=1,
                        help="Test set to use (1: Harvard, 2: RealSense)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to save results")
    
    # Visualization parameters
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions")
    parser.add_argument("--num_visualizations", type=int, default=5,
                        help="Number of samples to visualize")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    main(args)
