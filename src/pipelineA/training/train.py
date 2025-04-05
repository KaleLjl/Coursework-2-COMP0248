import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys # Added for verification exit

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_DATA_DIR, TRAIN_SEQUENCES, VALIDATION_FRAMES, TEST_FRAMES, # Use new frame lists
    POINT_CLOUD_PARAMS, MODEL_PARAMS, TRAIN_PARAMS, AUGMENTATION_PARAMS,
    WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR
)
from models.classifier import get_model
from models.utils import (
    save_checkpoint, load_checkpoint, get_lr, compute_metrics, 
    plot_confusion_matrix, plot_metrics, count_parameters, set_seed
)
from data_processing.dataset import create_data_loaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use
        
    Returns:
        tuple: (loss, metrics) where metrics is a dictionary of metrics
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    for batch_idx, data in enumerate(train_loader):
        # Get inputs and labels
        points = data['points'].to(device)
        labels = data['label'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient Clipping
        if TRAIN_PARAMS.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_PARAMS['gradient_clip'])
            
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        
        # Track predictions
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores))
    
    # Average loss
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, metrics

def validate(model, val_loader, criterion, device):
    """Validate the model.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (torch.utils.data.DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (loss, metrics) where metrics is a dictionary of metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Get inputs and labels
            points = data['points'].to(device)
            labels = data['label'].to(device)
            
            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            # Track loss
            running_loss += loss.item()
            
            # Track predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
    
    # Calculate metrics
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores))
    
    # Average loss
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, num_epochs, checkpoint_dir, log_dir):
    """Train the model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use
        num_epochs (int): Number of epochs to train
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory to save logs
        
    Returns:
        tuple: (model, train_losses, val_losses, train_metrics, val_metrics)
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize variables for tracking metrics
    train_losses = []
    val_losses = []
    train_metrics_list = []
    val_metrics_list = []
    
    # Early stopping parameters
    best_val_f1 = 0.0
    early_stopping_patience = TRAIN_PARAMS.get('early_stopping_patience', 15)
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"F1-Score: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"F1-Score: {val_metrics['f1']:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            print(f"Learning rate: {get_lr(optimizer):.6f}")
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1-Score/train', train_metrics['f1'], epoch)
        writer.add_scalar('F1-Score/val', val_metrics['f1'], epoch)
        writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
        writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
        writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, val_metrics['accuracy'], checkpoint_path)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_path = os.path.join(checkpoint_dir, "model_best.pt")
            save_checkpoint(model, optimizer, epoch, val_metrics['accuracy'], best_model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"F1-Score did not improve for {early_stopping_counter} epochs.")
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    
    # Close TensorBoard writer
    writer.close()
    
    return model, train_losses, val_losses, train_metrics_list, val_metrics_list

def main(args):
    """Main function for training.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    # Updated to use new data_spec arguments and pass individual params
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=BASE_DATA_DIR,
        train_spec=TRAIN_SEQUENCES,
        val_spec=VALIDATION_FRAMES,
        test_spec=TEST_FRAMES,
        batch_size=args.batch_size,         # Pass individual args
        num_workers=args.num_workers,       # Pass individual args
        point_cloud_params=POINT_CLOUD_PARAMS, # Pass config params
        augmentation_params=AUGMENTATION_PARAMS if args.augment else None # Pass config params conditionally
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    else:
        print("Test samples: 0 (No test set specified)")

    # Create model using parameters from config
    model = get_model(
        model_type=args.model_type, # Keep model_type configurable via CLI
        num_classes=2,
        k=args.k,                   # Keep k configurable via CLI
        emb_dims=MODEL_PARAMS['emb_dims'],
        dropout=MODEL_PARAMS['dropout'],
        feature_dropout=MODEL_PARAMS.get('feature_dropout', 0.0) # Use get for backward compatibility
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
        lr=args.learning_rate, # Keep LR configurable via CLI
        weight_decay=TRAIN_PARAMS['weight_decay'] # Use weight_decay from config
    )
    
    # Define learning rate scheduler using parameters from config
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # Monitor validation loss
        factor=TRAIN_PARAMS['lr_scheduler_factor'],
        patience=TRAIN_PARAMS['lr_scheduler_patience'],
        verbose=True
    )
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(
        WEIGHTS_DIR, f"{args.model_type}_{timestamp}")
    log_dir = os.path.join(
        LOGS_DIR, f"{args.exp_name if args.exp_name else f'{args.model_type}_{timestamp}'}") # Use exp_name if provided

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
    
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train point cloud classifier")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="dgcnn",
                        choices=["dgcnn", "pointnet"],
                        help="Model type")
    parser.add_argument("--k", type=int, default=MODEL_PARAMS.get('k', 20), # Default from config
                        help="Number of nearest neighbors for DGCNN")
    # Removed emb_dims and dropout as they are now loaded from config.py
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=TRAIN_PARAMS.get('num_epochs', 50), # Default from config
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=TRAIN_PARAMS.get('batch_size', 16), # Default from config
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=TRAIN_PARAMS.get('learning_rate', 0.001), # Default from config
                        help="Learning rate")
    # Removed weight_decay, lr_scheduler_patience, lr_scheduler_factor, train_val_split
    
    # Data parameters
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional name for the experiment run")

    args = parser.parse_args()
    main(args)
