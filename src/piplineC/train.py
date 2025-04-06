import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from utils import collate_fn


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Args:
            model: TableSegmentationModel
            train_dataset: DepthTableDataset for training
            val_dataset: DepthTableDataset for validation
            config: Training configuration dictionary
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            collate_fn=collate_fn
        )
        
        # Loss function with class weighting if needed
        if config.get("use_class_weights", False):
            # Calculate class weights from training data
            label_counts = self._calculate_class_distribution(train_dataset)
            weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
            weights = weights / weights.sum() * len(weights)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )
        elif config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config["learning_rate"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"]
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.config = config
        self.best_val_iou = 0.0
        self.start_epoch = 0
        
        # Setup logging
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(project=config["project_name"], config=config)
            wandb.watch(model)
    
    def _calculate_class_distribution(self, dataset):
        """Calculate distribution of classes in the dataset"""
        label_counts = np.zeros(2)  # Binary classification
        
        for i in range(len(dataset)):
            labels = dataset[i]['labels'].numpy()
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label < 2:  # Ensure label is valid
                    label_counts[label] += count
                    
        return label_counts
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_iou = 0.0
            train_acc = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            for batch in pbar:
                # Move data to device
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(points)
                
                # Reshape for loss calculation
                B, N, C = logits.shape
                loss = self.criterion(logits.view(B*N, C), labels.view(B*N))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                preds = torch.argmax(logits, dim=2)
                batch_iou = self._calculate_iou(preds, labels)
                batch_acc = (preds == labels).float().mean().item()
                
                train_loss += loss.item()
                train_iou += batch_iou
                train_acc += batch_acc
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{batch_iou:.4f}",
                    'acc': f"{batch_acc:.4f}"
                })
            
            # Calculate epoch metrics
            train_loss /= len(self.train_loader)
            train_iou /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            
            # Validation phase
            val_loss, val_iou, val_acc = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - " + 
                  f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, " +
                  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config["save_freq"] == 0:
                self.save_checkpoint(epoch)
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(points)
                
                # Calculate loss
                B, N, C = logits.shape
                loss = self.criterion(logits.view(B*N, C), labels.view(B*N))
                
                # Calculate metrics
                preds = torch.argmax(logits, dim=2)
                batch_iou = self._calculate_iou(preds, labels)
                batch_acc = (preds == labels).float().mean().item()
                
                val_loss += loss.item()
                val_iou += batch_iou
                val_acc += batch_acc
        
        # Calculate average metrics
        val_loss /= len(self.val_loader)
        val_iou /= len(self.val_loader)
        val_acc /= len(self.val_loader)
        
        return val_loss, val_iou, val_acc
    
    def _calculate_iou(self, preds, targets):
        """Calculate IoU for binary segmentation"""
        # Table class (1) IoU
        intersection = ((preds == 1) & (targets == 1)).float().sum().item()
        union = ((preds == 1) | (targets == 1)).float().sum().item()
        table_iou = intersection / (union + 1e-10)
        
        # Background class (0) IoU
        intersection = ((preds == 0) & (targets == 0)).float().sum().item()
        union = ((preds == 0) | (targets == 0)).float().sum().item()
        bg_iou = intersection / (union + 1e-10)
        
        # Mean IoU
        mean_iou = (table_iou + bg_iou) / 2
        
        return mean_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou,
            'config': self.config
        }
        
        if not os.path.exists(self.config["checkpoint_dir"]):
            os.makedirs(self.config["checkpoint_dir"])
        
        checkpoint_path = os.path.join(
            self.config["checkpoint_dir"], 
            f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.config["checkpoint_dir"], 
                "best_model.pth"
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with IoU: {self.best_val_iou:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_iou = checkpoint['best_val_iou']
        
        print(f"Loaded checkpoint from epoch {self.start_epoch}")
