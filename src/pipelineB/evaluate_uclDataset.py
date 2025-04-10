"""
This script 
  - runs the model on the UCL dataset
  - evaluates and saves the model performance metrics

References:
  - Provided lab materials.

Requirements:
  - Change the path configurations if needed.
"""
# ========== Imports ========================
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torchvision import models
# Custom imports
from dataloader import EstimatedDepthDataset, get_transform
from ResNet18_classification_model import modify_resnet18_input_channels

# ======== Config ===========================
# Please chanege the paths according to new directory structure if needed
DATA_PATH = "./data/pipelineB_data/Test_Data_2"
CSV_PATH = os.path.join(DATA_PATH, "labels.csv")
WEIGHT_PATH = "./weights/pipelineB/PRETRAINR18_15_32/resnet18_best.pth"
RESULTS_PATH = "./results/pipelineB/PRETRAINR18_15_32"
BATCH_SIZE = 32
INPUT_CHANNELS = 1
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load model =======================
model = models.resnet18(pretrained=False)
model = modify_resnet18_input_channels(model, pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device)
# Set model to evaluation mode
model.eval()

# ======== Load data ========================
test_dataset = EstimatedDepthDataset(DATA_PATH, transforms=get_transform(augment=False))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======== Criterion =====
criterion = nn.CrossEntropyLoss()

# ======== Evaluation Loop ==================
all_preds, all_labels = [], []
total_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ======== Metrics ==========================
avg_loss = total_loss / len(test_dataset)
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
cls_report = classification_report(all_labels, all_preds, digits=4)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n===== Evaluation Results on UCL Dataset =====")
print(f"Loss:      {avg_loss:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nClassification Report:\n", cls_report)
print("Confusion Matrix:\n", conf_matrix)

# Clear old results
r_path = os.path.join(RESULTS_PATH, "eval_report.txt")
if os.path.exists(r_path):
    print(f"Removing old metrics file: {r_path}")
    os.remove(r_path)

# Save to Text File
with open(os.path.join(RESULTS_PATH, "eval_report.txt"), "w") as f:
    f.write("===== Evaluation Results on UCL Dataset =====\n")
    f.write(f"Loss:      {avg_loss:.4f}\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(cls_report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")