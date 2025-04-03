import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch
        accuracy (float): Validation accuracy
        path (str): Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        path (str): Path to checkpoint
        
    Returns:
        tuple: (model, optimizer, epoch, accuracy)
    """
    # Check if checkpoint exists
    if not os.path.exists(path):
        print(f"Checkpoint {path} does not exist.")
        return model, optimizer, 0, 0.0
    
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"Checkpoint loaded from {path}, epoch {epoch}, accuracy {accuracy:.4f}")
    
    return model, optimizer, epoch, accuracy

def get_lr(optimizer):
    """Get current learning rate from optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_metrics(y_true, y_pred, y_score=None):
    """Compute classification metrics.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        y_score (numpy.ndarray, optional): Predicted scores for positive class
        
    Returns:
        dict: Dictionary of metrics
    """
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    
    # Compute AUC-ROC if scores are provided
    auc_roc = None
    if y_score is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_score)
        except Exception:
            # If there's only one class in y_true, ROC AUC score is not defined
            pass
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', path=None):
    """Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list, optional): List of class names
        title (str): Plot title
        path (str, optional): Path to save plot
    """
    if class_names is None:
        class_names = ['No Table', 'Table']
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    
    # Show plot
    plt.show()

def plot_metrics(train_metrics, val_metrics, metric_name, title=None, path=None):
    """Plot train and validation metrics over epochs.
    
    Args:
        train_metrics (list): Training metrics
        val_metrics (list): Validation metrics
        metric_name (str): Name of the metric
        title (str, optional): Plot title
        path (str, optional): Path to save plot
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot metrics
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.title(title or f'Training and Validation {metric_name.capitalize()}')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    
    # Show plot
    plt.show()

def count_parameters(model):
    """Count number of trainable parameters in a model.
    
    Args:
        model (nn.Module): Model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
