"""
This script 
  - loads a ResNet18 model,
  - trains it on a dataset of depth images,
  - evaluates its performance on a validation set.

References:
  - Provided lab materials.

Requirements:
  - Change the path configurations if needed.
"""
# ========== Imports ==========================
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# Custom imports
from dataloader import EstimatedDepthDataset, get_transform


# ========== Resnet Input Modification ========
def modify_resnet18_input_channels(model, pretrained=True):
    if pretrained:
        weight = model.conv1.weight.data
        new_weight = weight.mean(dim=1, keepdim=True)  # Averaged RGB → 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = new_weight
    else:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# ========== Fine Tuning ======================
def fine_tuning(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last block and classifier
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    # # Freeze convolution layers, so that we only train on classifier part
    # for param in model.features.parameters():
    #     param.requires_grad = False

# ========== Training and Validation ==========
def train_resnet_model(data_base_path, 
                       use_pretrained=True, 
                       num_epochs=20, 
                       batch_size=16, 
                       metrics_save_dir="results/pipelineB", 
                       weights_save_dir="weights/pipelineB"):
    
    os.makedirs(metrics_save_dir, exist_ok=True)
    os.makedirs(weights_save_dir, exist_ok=True)

    # Clear old results
    metrics_path = os.path.join(metrics_save_dir, "train_metrics.csv")
    if os.path.exists(metrics_path):
        print(f"Removing old metrics file: {metrics_path}")
        os.remove(metrics_path)

    for filename in ["resnet18_last.pth", "resnet18_best.pth"]:
        filepath = os.path.join(weights_save_dir, filename)
        if os.path.exists(filepath):
            print(f"Removing old weight file: {filepath}")
            os.remove(filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and adjust input and output
    model = models.resnet18(pretrained=use_pretrained)
    model = modify_resnet18_input_channels(model, pretrained=use_pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if use_pretrained:
        fine_tuning(model)

    model.to(device)

    # print(model)

    # Datasets and loaders
    train_dataset = EstimatedDepthDataset(os.path.join(data_base_path, "Training_Data"), transforms=get_transform(augment=True))
    val_dataset = EstimatedDepthDataset(os.path.join(data_base_path, "Test_Data_1"), transforms=get_transform(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    best_val_loss = float("inf")

    # Logging
    metrics_path = os.path.join(metrics_save_dir, "train_metrics.csv")
    write_header = not os.path.exists(metrics_path)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss, y_true, y_pred = 0.0, [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        # Compute training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred)
        train_recall = recall_score(y_true, y_pred)

        # Validation
        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                y_pred.extend(outputs.argmax(1).cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Compute validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred)
        val_recall = recall_score(y_true, y_pred)

        # Log epoch results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Save the most recent model
        torch.save(model.state_dict(), os.path.join(weights_save_dir, "resnet18_last.pth"))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_save_dir, "resnet18_best.pth"))

        # Save metrics
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss',
                    'train_acc', 'val_acc',
                    'train_f1', 'val_f1',
                    'train_precision', 'val_precision',
                    'train_recall', 'val_recall'
                ])
                write_header = False
            writer.writerow([
                epoch + 1,
                train_loss, val_loss,
                train_acc, val_acc,
                train_f1, val_f1,
                train_precision, val_precision,
                train_recall, val_recall
            ])

    print("Training complete. Models and metrics saved.")


if __name__ == "__main__":

    IF_PRETRAIN = True  # Set to False if train from scratch
    NUM_EPOCHS = 20
    BATCH_SIZE = 32

    if IF_PRETRAIN:
        print("Using pretrained weights.")
        train_resnet_model(
        data_base_path="./data/pipelineB_data/",
        use_pretrained=IF_PRETRAIN,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        metrics_save_dir="./results/pipelineB/PRETRAINR18_15_32",
        weights_save_dir="./weights/pipelineB/PRETRAINR18_15_32")
    else:
        print("Training from scratch.")
        train_resnet_model(
        data_base_path="./data/pipelineB_data/",
        use_pretrained=IF_PRETRAIN,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        metrics_save_dir="./results/pipelineB/TRAIN",
        weights_save_dir="./weights/pipelineB/TRAIN")
    

