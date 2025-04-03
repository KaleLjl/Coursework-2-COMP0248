import os
import numpy as np
import pickle
import glob
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import cv2

# Add the directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.depth_to_pointcloud import create_pointcloud_from_depth
from data_processing.preprocessing import preprocess_point_cloud
from config import POINT_CLOUD_PARAMS, AUGMENTATION_PARAMS

class TableDataset(Dataset):
    """Dataset for table detection from point clouds."""
    
    def __init__(self, data_root, sequences, transform=None, augment=False, 
                 mode='train', point_cloud_params=None, augmentation_params=None):
        """Initialize the dataset.
        
        Args:
            data_root (str): Root directory of the dataset
            sequences (dict): Dictionary of sequences to include. Format:
                {sequence_name: [sub_sequence_1, sub_sequence_2, ...]}
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to apply data augmentation
            mode (str): Dataset mode: 'train', 'val', or 'test'
            point_cloud_params (dict, optional): Parameters for point cloud processing
            augmentation_params (dict, optional): Parameters for data augmentation
        """
        self.data_root = Path(data_root)
        self.sequences = sequences
        self.transform = transform
        self.augment = augment and mode == 'train'  # Only augment during training
        self.mode = mode
        
        # Use default parameters if not provided
        self.point_cloud_params = point_cloud_params or POINT_CLOUD_PARAMS
        self.augmentation_params = augmentation_params or AUGMENTATION_PARAMS
        
        # List to store all data samples
        self.samples = []
        
        # Load all sample paths
        self._load_samples()
    
    def _load_samples(self):
        """Load all sample paths and labels."""
        for sequence_name, sub_sequences in self.sequences.items():
            for sub_sequence in sub_sequences:
                # Construct the paths
                sequence_dir = self.data_root / sequence_name / sub_sequence
                
                # Skip if directory doesn't exist
                if not sequence_dir.exists():
                    print(f"Warning: {sequence_dir} does not exist. Skipping.")
                    continue
                
                # Check if this is a raw depth sequence (harvard_tea_2)
                use_raw_depth = ('harvard_tea_2' in str(sequence_dir))
                
                # Get intrinsics path
                intrinsics_path = sequence_dir / "intrinsics.txt"
                if not intrinsics_path.exists():
                    print(f"Warning: No intrinsics found at {intrinsics_path}. Skipping.")
                    continue
                
                # Determine depth directory
                depth_dir_name = "depth" if use_raw_depth else "depthTSDF"
                depth_dir = sequence_dir / depth_dir_name
                if not depth_dir.exists():
                    print(f"Warning: No depth directory found at {depth_dir}. Skipping.")
                    continue
                
                # Get all depth files
                depth_files = sorted(glob.glob(str(depth_dir / "*")))
                
                # Get image directory (for visualization)
                image_dir = sequence_dir / "image"
                
                # Get label path
                label_path = sequence_dir / "labels" / "tabletop_labels.dat"
                
                # Load labels if they exist, otherwise all frames are negative samples
                if label_path.exists():
                    with open(label_path, 'rb') as f:
                        tabletop_labels = pickle.load(f)
                else:
                    # If no labels, set all frames as no-table (0)
                    tabletop_labels = [[]] * len(depth_files)
                
                # Make sure we have the same number of depth files and labels
                if len(depth_files) != len(tabletop_labels):
                    min_len = min(len(depth_files), len(tabletop_labels))
                    depth_files = depth_files[:min_len]
                    tabletop_labels = tabletop_labels[:min_len]
                
                # Get image files if available
                if image_dir.exists():
                    image_files = sorted(glob.glob(str(image_dir / "*")))
                    if len(image_files) != len(depth_files):
                        min_len = min(len(depth_files), len(image_files))
                        depth_files = depth_files[:min_len]
                        image_files = image_files[:min_len]
                        tabletop_labels = tabletop_labels[:min_len]
                else:
                    image_files = [None] * len(depth_files)
                
                # Add samples to the list
                for depth_file, image_file, label in zip(depth_files, image_files, tabletop_labels):
                    # Convert label to binary: 1 if table polygons exist, 0 otherwise
                    binary_label = 1 if len(label) > 0 else 0
                    
                    self.samples.append({
                        'depth_file': depth_file,
                        'image_file': image_file,
                        'intrinsics_path': str(intrinsics_path),
                        'use_raw_depth': use_raw_depth,
                        'label': binary_label,
                        'original_label': label,  # Store original label for visualization
                        'sequence': sequence_name,
                        'sub_sequence': sub_sequence
                    })
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: A dictionary containing:
                - points (torch.Tensor): Point cloud (Nx3)
                - label (int): Binary label (0 or 1)
                - metadata (dict): Additional information about the sample
        """
        sample = self.samples[idx]
        
        # Get sample paths
        depth_file = sample['depth_file']
        image_file = sample['image_file']
        intrinsics_path = sample['intrinsics_path']
        use_raw_depth = sample['use_raw_depth']
        
        # Extract point cloud
        try:
            # Create point cloud from depth map
            points = create_pointcloud_from_depth(
                depth_file, 
                intrinsics_path, 
                use_raw_depth=use_raw_depth,
                min_depth=self.point_cloud_params['min_depth'],
                max_depth=self.point_cloud_params['max_depth']
            )
            
            # Preprocess point cloud
            points, _ = preprocess_point_cloud(
                points, 
                normalize=self.point_cloud_params['normalize'],
                num_points=self.point_cloud_params['num_points'],
                sampling_method=self.point_cloud_params['sampling_method'],
                augment=self.augment,
                augmentation_params=self.augmentation_params if self.augment else None
            )
            
        except Exception as e:
            # If there's an error, return a default point cloud and log the error
            print(f"Error processing {depth_file}: {e}")
            points = np.zeros((self.point_cloud_params['num_points'], 3), dtype=np.float32)
        
        # Convert to torch tensor
        points_tensor = torch.from_numpy(points).float()
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        # Apply any additional transforms
        if self.transform is not None:
            points_tensor = self.transform(points_tensor)
        
        # Return a dictionary
        return {
            'points': points_tensor,
            'label': label_tensor,
            'metadata': {
                'depth_file': depth_file,
                'image_file': image_file,
                'sequence': sample['sequence'],
                'sub_sequence': sample['sub_sequence']
            }
        }

def collate_fn(batch):
    """Custom collate function to handle variable-sized metadata.
    
    Args:
        batch (list): List of samples returned by __getitem__
        
    Returns:
        dict: Collated batch with points, labels, and metadata
    """
    # Extract points and labels
    points = torch.stack([item['points'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Collect metadata as a list (do not try to stack)
    metadata = [item['metadata'] for item in batch]
    
    return {
        'points': points,
        'label': labels,
        'metadata': metadata
    }

def create_data_loaders(data_root, train_sequences, test_sequences, 
                        batch_size=16, num_workers=4, point_cloud_params=None,
                        augmentation_params=None, train_val_split=0.8):
    """Create data loaders for training, validation, and testing.
    
    Args:
        data_root (str): Root directory of the dataset
        train_sequences (dict): Dictionary of training sequences
        test_sequences (dict): Dictionary of test sequences
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        point_cloud_params (dict, optional): Parameters for point cloud processing
        augmentation_params (dict, optional): Parameters for data augmentation
        train_val_split (float): Fraction of training data to use for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create the training dataset
    train_dataset = TableDataset(
        data_root=data_root,
        sequences=train_sequences,
        augment=True,
        mode='train',
        point_cloud_params=point_cloud_params,
        augmentation_params=augmentation_params
    )
    
    # Split into training and validation
    train_size = int(train_val_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override the augment and mode for validation dataset
    val_dataset.dataset.augment = False
    val_dataset.dataset.mode = 'val'
    
    # Create the test dataset
    test_dataset = TableDataset(
        data_root=data_root,
        sequences=test_sequences,
        augment=False,
        mode='test',
        point_cloud_params=point_cloud_params,
        augmentation_params=None  # No augmentation for test
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader
