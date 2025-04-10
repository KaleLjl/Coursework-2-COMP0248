"""
This script 
  - loads the dataset
  - applies transformations

References:
  - Provided lab materials.
"""
# ========== Imports ================
import os
import glob
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

# ========== Main ===================
class EstimatedDepthDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (str): Root directory containing subfolders with depth npy and labels.csv
            transforms (callable, optional): Optional transform to apply on a sample
        """
        self.transforms = transforms
        self.samples = []

        # Traverse subdirectories
        subdirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]

        for subdir in subdirs:
            csv_path = os.path.join(subdir, "labels.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: No labels.csv found in {subdir}, skipping.")
                continue

            label_df = pd.read_csv(csv_path, dtype={'filename': str})

            for _, row in label_df.iterrows():
                filename = row['filename']
                label = int(row['label'])

                depth_path = os.path.join(subdir, filename + ".npy")
                if os.path.exists(depth_path):
                    self.samples.append((depth_path, label))
                else:
                    print(f"Missing depth file: {depth_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        depth_path, label = self.samples[idx]
        depth = np.load(depth_path).astype(np.float32)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # Shape: [1, H, W]

        if self.transforms:
            depth_tensor = self.transforms(depth_tensor)

        return depth_tensor, torch.tensor(label, dtype=torch.long)

# ========== Transforms =============
def get_transform(augment=False, size=(224, 224)):
    transforms = []
    if augment:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation(15))
    transforms.append(T.Resize(size))
    return T.Compose(transforms)

