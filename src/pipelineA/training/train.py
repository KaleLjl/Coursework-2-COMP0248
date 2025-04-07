import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
# import argparse # Removed argparse import
import sys # Added for verification exit

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_DATA_DIR, TRAIN_SEQUENCES, VALIDATION_FRAMES, TEST_FRAMES, # Use new frame lists
    POINT_CLOUD_PARAMS, MODEL_PARAMS, TRAIN_PARAMS, AUGMENTATION_PARAMS,
    WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR,
    # Add imports for parameters previously passed via CLI
    SEED, DEVICE, NUM_WORKERS, EXP_NAME, AUGMENT
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

def main():
    """Main function for training. Reads all configuration from config.py."""
    # Set random seed for reproducibility
    set_seed(SEED) # Use SEED from config

    # Set device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu") # Use DEVICE from config
    print(f"Using device: {device}")

    # Create data loaders using parameters from config
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=BASE_DATA_DIR,
        train_spec=TRAIN_SEQUENCES,
        val_spec=VALIDATION_FRAMES,
        test_spec=TEST_FRAMES,
        batch_size=TRAIN_PARAMS['batch_size'], # Use config value
        num_workers=NUM_WORKERS,               # Use config value
        point_cloud_params=POINT_CLOUD_PARAMS,
        augmentation_params=AUGMENTATION_PARAMS if AUGMENT else None # Use config value
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    else:
        print("Test samples: 0 (No test set specified)")

    # Create model using parameters from config
    model = get_model(
        model_type=MODEL_PARAMS['model_type'], # Use config value
        num_classes=2,
        k=MODEL_PARAMS.get('k', 20),           # Use config value
        emb_dims=MODEL_PARAMS['emb_dims'],
        dropout=MODEL_PARAMS['dropout'],
        feature_dropout=MODEL_PARAMS.get('feature_dropout', 0.0)
    )

    # Print model info
    num_params = count_parameters(model)
    print(f"Model: {MODEL_PARAMS['model_type']}") # Use config value
    print(f"Number of parameters: {num_params}")

    # Move model to device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer using parameters from config
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_PARAMS['learning_rate'], # Use config value
        weight_decay=TRAIN_PARAMS['weight_decay']
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
        WEIGHTS_DIR, f"{MODEL_PARAMS['model_type']}_{timestamp}") # Use config value
    log_dir = os.path.join(
        LOGS_DIR, f"{EXP_NAME if EXP_NAME else f'{MODEL_PARAMS['model_type']}_{timestamp}'}") # Use config value

    # Train model
    model, train_losses, val_losses, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=TRAIN_PARAMS['num_epochs'], # Use config value
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    # Plot training curves
    plot_metrics(
        [m['accuracy'] for m in train_metrics],
        [m['accuracy'] for m in val_metrics],
        'accuracy',
        path=os.path.join(RESULTS_DIR, f"{MODEL_PARAMS['model_type']}_{timestamp}_accuracy.png") # Use config value
    )

    plot_metrics(
        [m['f1'] for m in train_metrics],
        [m['f1'] for m in val_metrics],
        'f1-score',
        path=os.path.join(RESULTS_DIR, f"{MODEL_PARAMS['model_type']}_{timestamp}_f1.png") # Use config value
    )

    plot_metrics(
        train_losses,
        val_losses,
        'loss',
        path=os.path.join(RESULTS_DIR, f"{MODEL_PARAMS['model_type']}_{timestamp}_loss.png") # Use config value
    )

    print("Training completed.")

if __name__ == "__main__":
    # Removed argparse setup
    main() # Call main directly
