import os
import numpy as np
import pickle
import glob
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import cv2
import itertools
from collections import defaultdict

# Add the directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.depth_to_pointcloud import create_pointcloud_from_depth
from data_processing.preprocessing import preprocess_point_cloud
# Import specific config variables needed
from config import (
    POINT_CLOUD_PARAMS, AUGMENTATION_PARAMS, BASE_DATA_DIR,
    TRAIN_SEQUENCES, VALIDATION_FRAMES, TEST_FRAMES
)

class TableDataset(Dataset):
    """Dataset for table detection from point clouds."""

    def __init__(self, data_root, data_spec, transform=None, augment=False,
                 mode='train', point_cloud_params=None, augmentation_params=None):
        """Initialize the dataset.

        Args:
            data_root (str): Root directory of the dataset.
            data_spec (dict or list): Specification of data to load.
                - If dict: Dictionary of sequences {seq: [sub_seqs]} (for training).
                - If list: List of specific frame identifiers (relative paths from
                  data_root, e.g., 'harvard_c5/hv_c5_1/depth/frame-....png')
                  (for validation/test).
            transform (callable, optional): Optional transform for point clouds.
            augment (bool): Whether to apply data augmentation.
            mode (str): Dataset mode: 'train', 'val', or 'test'.
            point_cloud_params (dict, optional): Parameters for point cloud processing.
            augmentation_params (dict, optional): Parameters for data augmentation.
        """
        self.data_root = Path(data_root)
        self.data_spec = data_spec
        self.transform = transform
        self.augment = augment and mode == 'train' # Only augment during training
        self.mode = mode
        
        # Use default parameters if not provided
        self.point_cloud_params = point_cloud_params or POINT_CLOUD_PARAMS
        self.augmentation_params = augmentation_params or AUGMENTATION_PARAMS
        
        # List to store all data samples
        self.samples = []
        
        # Load and filter samples based on data_spec
        self._load_and_filter_samples()

    def _load_and_filter_samples(self):
        """Load all potential samples and filter based on data_spec."""
        all_potential_samples = []
        sequences_to_scan = {}

        # Determine which sequences need to be scanned
        if isinstance(self.data_spec, dict):
            sequences_to_scan = self.data_spec
            target_frame_ids = None # Not filtering by frame ID for training
        elif isinstance(self.data_spec, list):
            target_frame_ids = set(self.data_spec)
            # Parse frame IDs to find unique sequences to scan
            sequences_to_scan = defaultdict(set)
            for frame_id in target_frame_ids:
                try:
                    parts = Path(frame_id).parts
                    # Assuming structure like: sequence_name/sub_sequence_name/depth_dir/frame...
                    if len(parts) >= 3:
                         sequence_name = parts[0]
                         sub_sequence_name = parts[1]
                         sequences_to_scan[sequence_name].add(sub_sequence_name)
                    else:
                         print(f"Warning: Could not parse sequence from frame ID: {frame_id}")
                except Exception as e:
                    print(f"Warning: Error parsing frame ID {frame_id}: {e}")
            # Convert set back to list for iteration
            sequences_to_scan = {k: list(v) for k, v in sequences_to_scan.items()}
            if not sequences_to_scan:
                 print("Warning: No sequences identified from frame list. Dataset will be empty.")
                 self.samples = []
                 return
        else:
            raise ValueError("data_spec must be a dictionary or a list.")

        print(f"Mode '{self.mode}': Scanning sequences: {list(sequences_to_scan.keys())}")

        # Iterate through sequences determined above
        for sequence_name, sub_sequences in sequences_to_scan.items():
            for sub_sequence in sub_sequences:
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

                    # Calculate relative path for frame ID matching
                    try:
                        # Construct path relative to data_root for consistent ID
                        relative_depth_path = str(Path(depth_file).relative_to(self.data_root))
                    except ValueError:
                        relative_depth_path = depth_file # Fallback if not relative

                    all_potential_samples.append({
                        'frame_id': relative_depth_path, # Use relative path as ID
                        'depth_file': depth_file,
                        'image_file': image_file,
                        'intrinsics_path': str(intrinsics_path),
                        'use_raw_depth': use_raw_depth,
                        'label': binary_label,
                        'original_label': label,
                        'sequence': sequence_name,
                        'sub_sequence': sub_sequence
                    })

        # Filter the loaded samples
        if target_frame_ids:
            # Keep only samples whose frame_id is in the target list
            self.samples = [
                s for s in all_potential_samples if s['frame_id'] in target_frame_ids
            ]
            # Sanity check: ensure we found all requested frames
            found_ids = {s['frame_id'] for s in self.samples}
            missing_ids = target_frame_ids - found_ids
            if missing_ids:
                 print(f"Warning: Mode '{self.mode}' - Could not find data for {len(missing_ids)} requested frame IDs:")
                 # Print a few examples
                 for i, missing_id in enumerate(itertools.islice(missing_ids, 5)):
                      print(f"  - {missing_id}")
                 if len(missing_ids) > 5:
                      print(f"  ... and {len(missing_ids) - 5} more.")

        else:
            # Keep all loaded samples (training mode)
            self.samples = all_potential_samples

        print(f"Mode '{self.mode}': Loaded {len(self.samples)} samples.")


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
                'sub_sequence': sample['sub_sequence'],
                'frame_id': sample['frame_id'] # Add frame_id for reference
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

def create_data_loaders(data_root=BASE_DATA_DIR,
                        train_spec=TRAIN_SEQUENCES,
                        val_spec=VALIDATION_FRAMES,
                        test_spec=TEST_FRAMES,
                        batch_size=16, num_workers=4,
                        point_cloud_params=None,
                        augmentation_params=None):
    """Create data loaders for training, validation, and testing.

    Uses data specifications (sequence dict for train, frame lists for val/test)
    loaded from config.

    Args:
        data_root (str): Root directory of the dataset.
        train_spec (dict): Specification for training data (usually sequence dict).
        val_spec (list): Specification for validation data (usually frame list).
        test_spec (list): Specification for test data (usually frame list).
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        point_cloud_params (dict, optional): Parameters for point cloud processing.
        augmentation_params (dict, optional): Parameters for data augmentation.

    Returns:
        tuple: (train_loader, val_loader, test_loader or None)
    """
    from torch.utils.data import DataLoader

    # Use provided params or defaults from config
    pc_params = point_cloud_params or POINT_CLOUD_PARAMS
    aug_params = augmentation_params or AUGMENTATION_PARAMS

    # Create the training dataset
    train_dataset = TableDataset(
        data_root=data_root,
        data_spec=train_spec,
        augment=True,
        mode='train',
        point_cloud_params=pc_params,
        augmentation_params=aug_params
    )

    # Create the validation dataset
    val_dataset = TableDataset(
        data_root=data_root,
        data_spec=val_spec,
        augment=False,
        mode='val',
        point_cloud_params=pc_params,
        augmentation_params=None # No augmentation for validation
    )

    # Create the test dataset
    test_dataset = TableDataset(
        data_root=data_root,
        data_spec=test_spec,
        augment=False,
        mode='test',
        point_cloud_params=pc_params,
        augmentation_params=None # No augmentation for test
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
        shuffle=True, # Shuffle validation data as requested
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffling for test
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader
