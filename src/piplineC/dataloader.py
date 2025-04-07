import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from utils import depth_to_pointcloud

class DepthTableDataset(Dataset):
    def __init__(self, depth_dir, label_dir, intrinsic_matrix, transform=None):
        """
        Args:
            depth_dir: Directory with depth images
            label_dir: Directory with segmentation labels
            intrinsic_matrix: Camera intrinsic matrix
            transform: Optional transform to apply
        """
        self.depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) 
                                   if f.endswith(('.png', '.exr'))])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                   if f.endswith('.png')])
        self.intrinsic_matrix = intrinsic_matrix
        self.transform = transform
        
        assert len(self.depth_paths) == len(self.label_paths), "Mismatch in dataset sizes"
        
    def __len__(self):
        return len(self.depth_paths)
    
    def __getitem__(self, idx):
        # Load depth image
        depth_path = self.depth_paths[idx]
        if depth_path.endswith('.exr'):
            import OpenEXR
            depth_img =  # load EXR file
        else:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # Convert mm to meters
        
        # Load label image
        label_img = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert depth to point cloud
        points = depth_to_pointcloud(depth_img, self.intrinsic_matrix)
        
        # Get valid point labels (corresponding to valid depth values)
        valid_mask = depth_img.flatten() > 0
        labels = label_img.flatten()[valid_mask]
        
        # Convert to torch tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        if self.transform:
            points_tensor = self.transform(points_tensor)
        
        return {
            'points': points_tensor,
            'labels': labels_tensor,
            'path': self.depth_paths[idx]
        }
