import os
import glob
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from utils import get_mask, get_intrinsics, get_polygon, depth_to_pointcloud, \
    MIT_SEQUENCES, HARVARD_SEQUENCES, DATA_CONFIG


class Sun3D(Dataset):
    """
    Base dataset class for Sun3D data.
    """
    def __init__(self, data_root, sequences, split='train', transform=None):
        """
        Initialize the Sun3D base dataset.

        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to include
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
            on a sample
        """
        self.data_root = data_root
        self.sequences = sequences
        self.split = split
        self.transform = transform

        # Sequences known to be all negative samples (no tables)
        self.all_negative_sequences = ["mit_gym_z_squash", "harvard_tea_2"]

        # Load all data (depth, intrinsics, annotations)
        self.data_list = self._load_data()

    def _load_data(self):
        """
        Load all data from the specified sequences.

        Returns:
            list: List of dictionaries containing data
        """
        data_list = []

        for sequence in self.sequences:
            sequence_path = os.path.join(self.data_root, sequence)

            # Find all subdirectories in the sequence folder
            subdirs = [d for d in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, d))]

            # Debug counter for total table annotations
            sequence_table_count = 0
            sequence_image_count = 0

            for subdir in subdirs:
                subdir_path = os.path.join(sequence_path, subdir)

                # Check for different depth directory names
                depth_dir = None
                possible_depth_dirs = ['depthTSDF', 'depth']
                for possible_dir in possible_depth_dirs:
                    if os.path.exists(os.path.join(subdir_path, possible_dir)):
                        depth_dir = os.path.join(subdir_path, possible_dir)
                        break

                labels_dir = os.path.join(subdir_path, 'labels')
                intrinsics_file = os.path.join(subdir_path, 'intrinsics.txt')

                # Skip if required directories don't exist
                if depth_dir is None:
                    continue

                # Read intrinsics
                intrinsics = None
                if os.path.exists(intrinsics_file):
                    intrinsics = get_intrinsics(intrinsics_file)

                # Read annotations if available
                annotations = {}
                labels_file = os.path.join(labels_dir, 'tabletop_labels.dat')

                # Check if sequence is in all_negative_sequences list or if labels file exists
                if sequence in self.all_negative_sequences:
                    # For sequences known to be all negative, use empty annotations
                    annotations = {}  # Empty annotations for all images
                elif os.path.exists(labels_file):
                    annotations = get_polygon(labels_file)

                # Count images with table annotations but don't print
                subdir_table_count = sum(1 for polygons in annotations.values() if polygons)
                sequence_table_count += subdir_table_count

                # Get all depth maps
                depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))

                sequence_image_count += len(depth_files)

                # Parse timestamps from filenames
                depth_timestamps = {self._get_timestamp_from_filename(os.path.basename(f)): f for f in depth_files}

                for depth_ts, depth_file in depth_timestamps.items():
                    depth_filename = os.path.basename(depth_file).split('.')[0]
                    image_annotations = annotations.get(depth_filename, [])
                    data = {
                        'depth_file': depth_file,
                        'intrinsics': intrinsics,
                        'annotations': image_annotations,
                        'depth_timestamp': depth_ts,
                        'sequence': sequence,
                        'subdir': subdir
                    }

                    data_list.append(data)

        return data_list

    def _get_timestamp_from_filename(self, filename):
        """
        Extract timestamp from filename.

        Args:
            filename (str): Filename

        Returns:
            float: Timestamp in seconds
        """
        try:
            parts = filename.split('-')
            if len(parts) >= 2:
                timestamp_str = parts[1].split('.')[0]
                return float(timestamp_str) / 1e6  # Convert to seconds
            return 0.0
        except:
            return 0.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get a data sample.

        Args:
            idx (int): Index

        Returns:
            dict: A data sample
        """
        data_list = self.data_list[idx]

        # Load depth map
        depth_map = cv2.imread(data_list['depth_file'], cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            depth_map = np.zeros((480, 640), dtype=np.uint16)

        # Convert depth map to meters if needed
        depth_map = depth_map.astype(np.float32)
        if depth_map.max() > 1000:  # If values are in millimeters
            depth_map /= 1000.0  # Convert to meters

        # Apply a simple median filter to reduce noise
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)

        # Get image shape
        image_shape = depth_map.shape

        # Get intrinsics - use default values if intrinsics are missing
        intrinsics = data_list['intrinsics']
        if intrinsics is None:
            # Use default intrinsics as fallback
            intrinsics = {
                'fx': 525.0,  # Default values from common RGBD cameras
                'fy': 525.0,
                'cx': 319.5,
                'cy': 239.5
            }

        # Get annotations
        annotations = data_list['annotations']

        # Get binary label and mask
        binary_label, binary_mask = get_mask(annotations, image_shape)

        # return in dictionary
        return {
            'depth_map': depth_map,
            'intrinsics': intrinsics,
            'annotations': annotations,
            'binary_label': binary_label,
            'binary_mask': binary_mask,
            'sequence': data_list['sequence'],
            'subdir': data_list['subdir']
        }


class SegmentationSubset(Subset):
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


class SegmentationDataset(Sun3D):
    """
    Dataset class for point cloud segmentation.
    """
    def __init__(self, data_root, sequences, split='train', transform=None, 
                 num_points=4096, use_height=True, random_sampling=True):
        """
        Initialize the point cloud segmentation dataset.

        Args:
            data_root (str): Root directory of the dataset
            sequences (list): List of sequences to include
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on point clouds
            num_points (int): Number of points to sample from the point cloud
            use_height (bool): Whether to use height as additional feature
            random_sampling (bool): Whether to use random sampling (True) or farthest point sampling (False)
        """
        super().__init__(data_root, sequences, split, transform)
        self.num_points = num_points
        self.use_height = use_height
        self.random_sampling = random_sampling

        # Class statistics
        self.class_counts = defaultdict(int)
        self.total_points = 0

        # Calculate class statistics
        self._calculate_class_statistics()

    def _calculate_class_statistics(self):
        """
        Calculate class statistics for the dataset.
        This is useful for balancing the loss function.
        """
        print("Calculating class statistics for point cloud segmentation dataset...")

        for idx in range(len(self.data_list)):
            try:
                # We can't assume sample already has point_cloud and point_labels
                # We need to get the base sample and create these fields
                base_sample = super().__getitem__(idx)
                binary_label = base_sample['binary_label']

                # Record class statistics
                if binary_label == 1:  # Has table
                    self.class_counts[1] += 1
                else:  # No table
                    self.class_counts[0] += 1
                self.total_points += 1
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

        # Print class statistics
        print("Class statistics:")
        for class_id, count in self.class_counts.items():
            print(f"  Class {class_id}: {count} samples ({count/self.total_points*100:.2f}%)")

    def __getitem__(self, idx):
        """
        Get a data sample.

        Args:
            idx (int): Index

        Returns:
            dict: A data sample including point cloud, point labels, RGB image, and metadata
        """
        # Get base sample from Sun3DBaseDataset
        sample = super().__getitem__(idx)

        # Check if the sample is valid
        if 'depth_map' not in sample or sample['depth_map'] is None:
            # If no depth map, create a simple dummy point cloud and labels
            print(f"Warning: Missing depth map for sample {idx}. Creating dummy point cloud.")
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
            binary_label = sample.get('binary_label', 0)
            point_labels = np.full(self.num_points, binary_label, dtype=np.int64)
        else:
            # Create point cloud from depth map
            depth_map = sample['depth_map']
            intrinsics = sample['intrinsics']
            binary_mask = sample['binary_mask']

            # Create point cloud from depth map
            point_cloud = depth_to_pointcloud(
                depth_map,
                intrinsics,
                subsample=True,
                num_points=self.num_points
            )

            # Initialize point-level labels
            point_labels = np.zeros(len(point_cloud), dtype=np.int64)

            # Get image dimensions
            height, width = depth_map.shape

            # Assign labels to each point
            for i, (x, y, z) in enumerate(point_cloud[:, :3]):
                if z <= 0:
                    continue

                # Project point back to image space
                u = int(x * intrinsics['fx'] / z + intrinsics['cx'])
                v = int(y * intrinsics['fy'] / z + intrinsics['cy'])

                # Check if projected point is in the image and has a valid mask value
                if 0 <= u < width and 0 <= v < height:
                    point_labels[i] = binary_mask[v, u]

        # Extract XYZ coordinates
        xyz = point_cloud[:, :3]

        # Get colors if available (assuming point_cloud has shape (N, 6) with RGB in last 3 columns)
        if point_cloud.shape[1] >= 6:
            rgb = point_cloud[:, 3:6]
        else:
            # If no colors, use default colors
            rgb = np.zeros((xyz.shape[0], 3))

        # Normalize point cloud to [0, 1]
        min_coords = np.min(xyz, axis=0)
        max_coords = np.max(xyz, axis=0)
        # Avoid division by zero
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0  # Avoid division by zero
        xyz_normalized = (xyz - min_coords) / (range_coords + 1e-8)

        # Add height as feature if requested
        if self.use_height:
            # Calculate height (y-coordinate) relative to the lowest point
            floor_height = np.min(xyz[:, 1])
            heights = xyz[:, 1] - floor_height
            # Normalize heights to [0, 1]
            max_height = np.max(heights)
            if max_height == 0:
                heights = np.zeros_like(heights)
            else:
                heights = heights / (max_height + 1e-8)
            heights = heights.reshape(-1, 1)

            # Combine features: XYZ, height, RGB
            point_features = np.concatenate([xyz_normalized, heights, rgb], axis=1)  # (N, 7)
        else:
            # Combine features: XYZ, RGB
            point_features = np.concatenate([xyz, rgb], axis=1)  # (N, 6)

        # Convert to torch tensors
        point_features = torch.tensor(point_features, dtype=torch.float32)
        point_labels = torch.tensor(point_labels, dtype=torch.long)

        # Apply transforms if provided
        if self.transform is not None:
            point_features = self.transform(point_features)

        # Create the final sample dictionary
        counts = torch.bincount(point_labels, minlength=2)  # [1, 2]
        # print(point_features.shape)
        pc_sample = {
            'point_features': point_features,
            'point_cloud': point_features,
            'point_labels': point_labels,
            'labels': point_labels,
            'label_counts': counts,
            'sequence': sample['sequence'],
            'subdir': sample['subdir']
        }

        return pc_sample

    def get_class_weights(self):
        """
        Calculate class weights inversely proportional to class frequencies.

        Returns:
            torch.Tensor: Class weights
        """
        if sum(self.class_counts.values()) == 0:
            return torch.ones(2)  # Default equal weights if no statistics

        # Get counts for each class, ensuring we have all classes
        counts = [self.class_counts.get(i, 1) for i in range(2)]

        # Calculate weights inversely proportional to class frequencies
        weights = 1.0 / np.array(counts)
        weights = weights / np.sum(weights)  # Normalize

        return torch.tensor(weights, dtype=torch.float32)


class PointCloudTransform:
    """
    Transformations for point clouds.
    """
    def __init__(self, mode='train'):
        """
        Initialize point cloud transforms.

        Args:
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode

    def __call__(self, points):
        """
        Apply transforms to the point cloud.

        Args:
            points (torch.Tensor): Input point cloud of shape (N, C)

        Returns:
            torch.Tensor: Transformed point cloud
        """
        if self.mode == 'train':
            # Data augmentation for training
            points = self._random_rotate(points)
            points = self._random_scale(points)
            points = self._random_translate(points)
            points = self._jitter(points)

        return points

    def _random_rotate(self, points):
        """
        Random rotation around the z-axis.

        Args:
            points (torch.Tensor): Input point cloud

        Returns:
            torch.Tensor: Rotated point cloud
        """
        theta = torch.FloatTensor(1).uniform_(0, 2 * np.pi)[0]
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Apply rotation only to XYZ coordinates
        rotated_coords = torch.matmul(points[:, :3], rotation_matrix)
        points = torch.cat([rotated_coords, points[:, 3:]], dim=1)

        return points

    def _random_scale(self, points):
        """
        Random scaling.

        Args:
            points (torch.Tensor): Input point cloud

        Returns:
            torch.Tensor: Scaled point cloud
        """
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2)[0]

        # Apply scaling only to XYZ coordinates
        scaled_coords = points[:, :3] * scale
        points = torch.cat([scaled_coords, points[:, 3:]], dim=1)

        return points

    def _random_translate(self, points):
        """
        Random translation.

        Args:
            points (torch.Tensor): Input point cloud

        Returns:
            torch.Tensor: Translated point cloud
        """
        translation = torch.FloatTensor(3).uniform_(-0.1, 0.1)

        # Apply translation only to XYZ coordinates
        translated_coords = points[:, :3] + translation
        points = torch.cat([translated_coords, points[:, 3:]], dim=1)

        return points

    def _jitter(self, points):
        """
        Add random jitter to point coordinates.

        Args:
            points (torch.Tensor): Input point cloud

        Returns:
            torch.Tensor: Jittered point cloud
        """
        noise = torch.FloatTensor(points.size()).uniform_(-0.01, 0.01)

        # Apply jitter only to XYZ coordinates
        jittered_coords = points[:, :3] + noise[:, :3]
        points = torch.cat([jittered_coords, points[:, 3:]], dim=1)

        return points


def get_dataloader(config, split='train', transform=None, test_set=1):
    """
    Get dataloader for Pipeline C.

    Args:
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply to the data
        test_set (int): Test set to use (1: Harvard, 2: RealSense)

    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified split
    """

    # Determine which sequences to use based on the split
    if split == 'test':
        sequences = HARVARD_SEQUENCES
        if test_set == 2:
            sequences = ["ucl"]
    else:
        sequences = MIT_SEQUENCES

    # Create dataset
    dataset = SegmentationDataset(
        data_root=config['data_root'],
        sequences=sequences,
        split=split,
        transform=transform,
        num_points=config['num_points'],
        random_sampling=(split == 'train')
    )

    # If split is 'train' or 'val', split the training data
    if split in ['train', 'val']:
        # Calculate the number of samples for training and validation
        train_val_split = 0.8  # Default 80% training, 20% validation
        if 'train_val_split' in config:
            train_val_split = config['train_val_split']

        num_samples = len(dataset)
        indices = np.arange(num_samples)

        # Set the random seed for reproducible splits
        np.random.seed(42)
        np.random.shuffle(indices)

        split_idx = int(num_samples * train_val_split)

        if split == 'train':
            subset_indices = indices[:split_idx]
        else:
            subset_indices = indices[split_idx:]

        # Create subset
        dataset = SegmentationSubset(dataset, subset_indices)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


def get_dataloaders(test_set=1):
    """
    Get all dataloaders for Pipeline C.

    Args:
        test_set (int): Test set to use (1: Harvard, 2: RealSense)

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Get transforms
    train_transform = PointCloudTransform('train')
    val_transform = PointCloudTransform('val')
    test_transform = PointCloudTransform('test')

    # Get dataloaders
    train_dataloader = get_dataloader(DATA_CONFIG, 'train', train_transform)
    val_dataloader = get_dataloader(DATA_CONFIG, 'val', val_transform)
    test_dataloader = get_dataloader(DATA_CONFIG, 'test', test_transform,
                                     test_set)

    return train_dataloader, val_dataloader, test_dataloader
