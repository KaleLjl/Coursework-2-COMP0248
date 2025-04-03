import os
import sys
import argparse
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DATA_DIR, TRAIN_SEQUENCES, TEST1_SEQUENCES, TEST2_SEQUENCES,
    POINT_CLOUD_PARAMS, MODEL_PARAMS, TRAIN_PARAMS, AUGMENTATION_PARAMS,
    WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR
)
from models.classifier import get_model
from models.utils import set_seed
from data_processing.dataset import create_data_loaders, TableDataset, collate_fn
from training.train import train_model
from training.evaluate import evaluate_model, analyze_results_by_sequence

def train(args):
    """Train a model.
    
    Args:
        args: Command line arguments
    """
    # Import necessary modules
    import torch.nn as nn
    import torch.optim as optim
    from datetime import datetime
    from models.utils import count_parameters, get_lr, plot_metrics, save_checkpoint
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=BASE_DATA_DIR,
        train_sequences=TRAIN_SEQUENCES,
        test_sequences=TEST1_SEQUENCES,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        point_cloud_params=POINT_CLOUD_PARAMS,
        augmentation_params=AUGMENTATION_PARAMS if args.augment else None,
        train_val_split=args.train_val_split
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = get_model(
        model_type=args.model_type,
        num_classes=2,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout
    )
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model: {args.model_type}")
    print(f"Number of parameters: {num_params}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        verbose=True
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(
        WEIGHTS_DIR, f"{args.model_type}_{timestamp}")
    log_dir = os.path.join(
        LOGS_DIR, f"{args.model_type}_{timestamp}")
    
    # Train model
    model, train_losses, val_losses, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Plot training curves
    plot_metrics(
        [m['accuracy'] for m in train_metrics],
        [m['accuracy'] for m in val_metrics],
        'accuracy',
        path=os.path.join(RESULTS_DIR, f"{args.model_type}_{timestamp}_accuracy.png")
    )
    
    plot_metrics(
        [m['f1'] for m in train_metrics],
        [m['f1'] for m in val_metrics],
        'f1-score',
        path=os.path.join(RESULTS_DIR, f"{args.model_type}_{timestamp}_f1.png")
    )
    
    plot_metrics(
        train_losses,
        val_losses,
        'loss',
        path=os.path.join(RESULTS_DIR, f"{args.model_type}_{timestamp}_loss.png")
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(model, optimizer, args.num_epochs, val_metrics[-1]['accuracy'], final_model_path)
    
    print("Training completed.")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print(f"Best model: {os.path.join(checkpoint_dir, 'model_best.pt')}")

def evaluate(args):
    """Evaluate a trained model.
    
    Args:
        args: Command line arguments
    """
    # Import necessary modules
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from data_processing.dataset import TableDataset
    from models.utils import load_checkpoint, plot_confusion_matrix
    
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
    for sequence, seq_metrics in sequence_metrics.items():
        print(f"\n{sequence}:")
        print(f"  Accuracy: {seq_metrics['accuracy']:.4f}")
        print(f"  Precision: {seq_metrics['precision']:.4f}")
        print(f"  Recall: {seq_metrics['recall']:.4f}")
        print(f"  F1-Score: {seq_metrics['f1']:.4f}")
    
    # Visualize some predictions if requested
    if args.visualize:
        from training.evaluate import visualize_predictions
        print("\nVisualizing predictions...")
        visualize_predictions(
            predictions=predictions,
            results_dir=results_dir,
            num_samples=args.num_visualizations
        )
    
    print("Evaluation completed.")

def main():
    """Main function."""
    # Create parser
    parser = argparse.ArgumentParser(description="Pipeline A: Table Detection from Point Clouds")
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    
    # Model parameters
    train_parser.add_argument("--model_type", type=str, default="dgcnn",
                        choices=["dgcnn", "pointnet"],
                        help="Model type")
    train_parser.add_argument("--k", type=int, default=20,
                        help="Number of nearest neighbors for DGCNN")
    train_parser.add_argument("--emb_dims", type=int, default=1024,
                        help="Embedding dimensions")
    train_parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    
    # Training parameters
    train_parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    train_parser.add_argument("--lr_scheduler_patience", type=int, default=5,
                        help="Learning rate scheduler patience")
    train_parser.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                        help="Learning rate scheduler factor")
    train_parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Train/validation split ratio")
    
    # Data parameters
    train_parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation")
    train_parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Other parameters
    train_parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    train_parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    # Evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    
    # Model parameters
    eval_parser.add_argument("--model_type", type=str, default="dgcnn",
                        choices=["dgcnn", "pointnet"],
                        help="Model type")
    eval_parser.add_argument("--k", type=int, default=20,
                        help="Number of nearest neighbors for DGCNN")
    eval_parser.add_argument("--emb_dims", type=int, default=1024,
                        help="Embedding dimensions")
    
    # Evaluation parameters
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    eval_parser.add_argument("--test_set", type=int, default=1,
                        help="Test set to use (1: Harvard, 2: RealSense)")
    eval_parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    eval_parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to save results")
    
    # Visualization parameters
    eval_parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions")
    eval_parser.add_argument("--num_visualizations", type=int, default=5,
                        help="Number of samples to visualize")
    
    # Other parameters
    eval_parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    eval_parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    eval_parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run based on mode
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
