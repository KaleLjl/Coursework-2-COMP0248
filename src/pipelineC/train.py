import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.pipelineC.model import DGCNN_Seg
from src.pipelineC.dataloader import get_dataloaders
from src.pipelineC.utils import compute_metrics
from src.pipelineC.evaluate import evaluate


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for segmentation.
    Dice loss optimizes the Dice coefficient (F1 score) directly.
    """
    def __init__(self, smooth=1.0, ignore_index=-1):
        """
        Initialize dice loss.

        Args:
            smooth (float): Smoothing term to avoid division by zero
            ignore_index (int): Index to ignore in the loss calculation
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Forward pass of dice loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N) or (B*N, C)
            targets (torch.Tensor): Ground truth labels of shape (B, N) or (B*N)

        Returns:
            torch.Tensor: Loss value
        """
        # Ensure inputs are properly shaped
        if logits.dim() == 3:  # (B, C, N)
            _, num_classes, _ = logits.size()
            logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
            logits = logits.view(-1, num_classes)  # (B*N, C)
            targets = targets.view(-1)  # (B*N)

        # Create one-hot encoding for targets
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Mask out ignored indices
        mask = (targets != self.ignore_index).float().unsqueeze(1)
        one_hot = one_hot * mask

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Multiply by mask to ignore certain indices
        probs = probs * mask

        # Calculate dice coefficient for each class
        numerator = 2 * (probs * one_hot).sum(0) + self.smooth
        denominator = probs.sum(0) + one_hot.sum(0) + self.smooth
        dice = numerator / denominator

        # Calculate mean dice loss
        loss = 1 - dice.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combines different loss functions with optional weights.
    """
    def __init__(self, losses, weights=None):
        """
        Initialize combined loss.

        Args:
            losses (list): List of loss functions
            weights (list, optional): Weights for each loss
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, logits, targets):
        """
        Forward pass of combined loss.

        Args:
            logits (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Combined loss value
        """
        combined_loss = 0.0
        for i, loss_fn in enumerate(self.losses):
            if isinstance(loss_fn, nn.CrossEntropyLoss):  # ce
                _, num_classes, _ = logits.size()
                logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
                logits = logits.view(-1, num_classes)  # (B*N, C)
                targets = targets.view(-1)  # (B*N)
            combined_loss += self.weights[i] * loss_fn(logits, targets)
        return combined_loss


class SegmentationLoss(nn.Module):
    """
    Loss function for point cloud segmentation.
    Combines different loss functions including cross entropy, focal loss, and dice loss.
    """
    def __init__(self, class_weights=None, ignore_index=-1, loss_weights=None):
        """
        Initialize segmentation loss.

        Args:
            class_weights (list or torch.Tensor, optional): Class weights for weighted loss
            ignore_index (int): Index to ignore in the loss calculation
            loss_weights (list, optional): Weights for combined loss
        """
        super(SegmentationLoss, self).__init__()
        self.ignore_index = ignore_index

        # Convert class_weights to tensor if it's a list
        if class_weights is not None:
            if isinstance(class_weights, list):
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                self.class_weights = class_weights
        else:
            self.class_weights = None

        # Note: We'll initialize the actual criterion in the forward method
        # to ensure it's on the correct device
        self.criterion = None
        self.loss_weights = loss_weights

    def forward(self, logits, targets):
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): Predicted logits of shape (B, C, N)
            targets (torch.Tensor): Ground truth labels of shape (B, N)

        Returns:
            torch.Tensor: Loss value
        """
        # Get device of input tensors
        device = logits.device

        # Move class_weights to the same device if needed
        if self.class_weights is not None and self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)

        # Create the loss function on first forward pass or if device changes
        if self.criterion is None:
            # Default: equally weighted CE and Dice loss
            weights = self.loss_weights if self.loss_weights else [0.5, 0.5]
            self.criterion = CombinedLoss(
                losses=[
                    nn.CrossEntropyLoss(weight=self.class_weights,
                                        ignore_index=self.ignore_index),
                    DiceLoss(ignore_index=self.ignore_index)
                ],
                weights=weights
            )

        # Compute the loss
        return self.criterion(logits, targets)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to use
        epoch (int): Current epoch

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/200 [Train]")

    for batch_idx, data in enumerate(pbar):
        # Get data
        inputs = data['point_features'].to(device)
        targets = data['point_labels'].to(device)
        counts = data['label_counts']

        class_weight = torch.sum(counts, dim=0)
        class_weight = torch.max(class_weight) / (class_weight + 1e-9)

        inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        criterion.class_weights = class_weight
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update loss
        total_loss += loss.item()

        # Calculate predictions
        preds = torch.argmax(outputs, dim=1)

        # Store predictions and targets for metrics calculation
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })

    # Compute average loss
    train_loss = total_loss / len(train_loader)

    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])

    train_metrics = compute_metrics(flat_targets, flat_preds)

    # Print summary to console
    print(f"\nTrain Epoch: {epoch+1} Summary:")
    print(f"  Loss:            {train_loss:.4f}")
    print(f"  Accuracy:        {train_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {train_metrics['mean_iou']:.4f}")
    print(f"  Background IoU:  {train_metrics['iou_background']:.4f}")
    print(f"  Table IoU:       {train_metrics['iou_table']:.4f}")
    print(f"  F1 Score:        {train_metrics.get('f1_weighted', 0.0):.4f}")

    return train_loss, train_metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        epoch (int): Current epoch

    Returns:
        tuple: (average validation loss, validation metrics)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/200 [Val]")

    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            # Get data
            inputs = data['point_features'].to(device)
            targets = data['point_labels'].to(device)
            counts = data['label_counts']

            class_weight = torch.sum(counts, dim=0)
            class_weight = torch.max(class_weight) / (class_weight + 1e-9)
            # print(f"class weight: {class_weight}")

            # Transpose input if needed - model expects (B, C, N) but data might be (B, N, C)
            if inputs.shape[1] == 5120:  # If second dimension is num_points, we need to transpose
                inputs = inputs.transpose(1, 2)  # (B, N, C) -> (B, C, N)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            criterion.class_weights = class_weight
            loss = criterion(outputs, targets)

            # Update loss
            val_loss += loss.item()

            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and targets for metrics calculation
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'avg_val_loss': f"{val_loss / (batch_idx + 1):.4f}"
            })

    # Compute average loss
    val_loss /= len(dataloader)

    # Compute metrics
    # Concatenate all predictions and targets
    flat_preds = np.concatenate([p.flatten() for p in all_preds])
    flat_targets = np.concatenate([t.flatten() for t in all_targets])

    val_metrics = compute_metrics(flat_targets, flat_preds)

    # Print summary to console for better visibility
    print(f"\nValidation Epoch: {epoch+1} Summary:")
    print(f"  Loss:            {val_loss:.4f}")
    print(f"  Accuracy:        {val_metrics['accuracy']:.4f}")
    print(f"  Mean IoU:        {val_metrics['mean_iou']:.4f}")
    print(f"  Table IoU:       {val_metrics['iou_table']:.4f}")
    print(f"  Background IoU:  {val_metrics['iou_background']:.4f}")
    print(f"  F1 Score:        {val_metrics.get('f1_weighted', 0.0):.4f}")

    return val_loss, val_metrics


def main():
    """
    Main function for training Pipeline C.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = DGCNN_Seg()
    model = model.to(device)

    loss_weights = [0.7, 0.3]

    criterion = SegmentationLoss(
        loss_weights=loss_weights
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=40,
        gamma=0.7
    )

    # Create checkpoint directory
    checkpoint_dir = "weights/pipelineC"
    os.makedirs("weights/pipelineC", exist_ok=True)

    # Create experiment directory
    experiment_dir = "results/pipelineC"
    os.makedirs(experiment_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    for epoch in range(200):
        # Train for one epoch
        _, _ = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # Update learning rate
        scheduler.step()

        # Update best model
        current_val_iou = val_metrics['mean_iou']
        if current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            best_val_loss = val_loss
            best_epoch = epoch

            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
            }, checkpoint_path)
            print(f"Best model saved at epoch {epoch+1} with IoU: {best_val_iou:.4f}")

            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            # Increment early stopping counter
            early_stopping_counter += 1

            # Check if early stopping criteria is met
            if early_stopping_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}. Best IoU: {best_val_iou:.4f} at epoch {best_epoch+1}")
                break

    # Train finished
    print("\nTraining finished!")
    print(f"Best validation performance at epoch {best_epoch+1}:")
    print(f"  IoU (Table): {best_val_iou:.4f}")
    print(f"  Loss: {best_val_loss:.4f}")

    # Load best model for testing
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Test best model
    test_output_dir = os.path.join(experiment_dir, 'test_results')
    test_loss, test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=test_output_dir
    )

    # Print final test results
    print("\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"  Background IoU: {test_metrics['iou_background']:.4f}")
    print(f"  Table IoU: {test_metrics['iou_table']:.4f}")
    print(f"  F1 Score: {test_metrics.get('f1_weighted', 0.0):.4f}")


if __name__ == '__main__':
    main()
