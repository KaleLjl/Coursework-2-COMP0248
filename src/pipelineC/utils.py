from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import cv2
import os
from tqdm import tqdm
import glob
import json


SEED = 42
ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "data"
WEIGHTS_DIR = ROOT_DIR / "weights"
RESULTS_DIR = ROOT_DIR / "results"
MIT_SEQUENCES = {
    "mit_32_d507": ["d507_2"],
    "mit_76_459": ["76-459b"],
    "mit_76_studyroom": ["76-1studyroom2"],
    "mit_gym_z_squash": ["gym_z_squash_scan1_oct_26_2012_erika"],
    "mit_lab_hj": ["lab_hj_tea_nov_2_2012_scan1_erika"]
}
HARVARD_SEQUENCES = {
    "harvard_c5": ["hv_c5_1"],
    "harvard_c6": ["hv_c6_1"],
    "harvard_c11": ["hv_c11_2"],
    "harvard_tea_2": ["hv_tea2_2"]
}
REAL_SENSE_SEQUENCES = {
    "real_sense_1": ["real_sense_1"],
    "real_sense_2": ["real_sense_2"]
}


def depth_to_pointcloud(depth_image, intrinsic_matrix):
    """Convert depth image to point cloud using camera intrinsics

    Args:
        depth_image: Numpy array of depth values
        intrinsic_matrix: 3x3 camera intrinsic matrix

    Returns:
        points: Nx3 numpy array of 3D points
    """
    # Get image dimensions
    height, width = depth_image.shape

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.flatten(), v.flatten()

    # Filter out invalid depth values
    valid_depth = depth_image.flatten() > 0
    z = depth_image.flatten()[valid_depth]
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]

    # Calculate 3D coordinates
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    # Combine into point cloud
    points = np.stack([x, y, z], axis=1)

    return points


def collate_fn(batch):
    """Custom collate function to handle variable point cloud sizes"""
    points = [item['points'] for item in batch]
    labels = [item['labels'] for item in batch]
    paths = [item['path'] for item in batch]

    # Pad point clouds to same size or use a different approach
    max_points = max(p.shape[0] for p in points)
    points_padded = []
    labels_padded = []

    for p, l in zip(points, labels):
        if p.shape[0] < max_points:
            padded_points = torch.zeros((max_points, 3), dtype=torch.float32)
            padded_labels = torch.zeros(max_points, dtype=torch.long)
            padded_points[:p.shape[0], :] = p
            padded_labels[:l.shape[0]] = l
            points_padded.append(padded_points)
            labels_padded.append(padded_labels)
        else:
            points_padded.append(p)
            labels_padded.append(l)

    return {
        'points': torch.stack(points_padded),
        'labels': torch.stack(labels_padded),
        'paths': paths
    }


def create_dataset_from_rgbd(
    rgb_dir,
    depth_dir,
    output_dir,
    intrinsic_matrix,
    annotation_file=None,
    table_height_threshold=None
):
    """
    Create a point cloud dataset with table segmentation from RGB-D data

    Args:
        rgb_dir: Directory with RGB images
        depth_dir: Directory with depth images
        output_dir: Output directory for segmented point clouds
        intrinsic_matrix: Camera intrinsic matrix
        annotation_file: Optional JSON file with table annotations
        table_height_threshold: Optional height threshold for table segmentation
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "point_clouds"), exist_ok=True)

    # Get file lists
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    assert len(rgb_files) == len(depth_files), "RGB and depth file counts don't match"

    # Load annotations if provided
    annotations = {}
    if annotation_file:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

    # Process each pair of images
    for i, (rgb_path, depth_path) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files))):
        # Get filenames
        filename = os.path.basename(rgb_path).split('.')[0]

        # Load images
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # Convert mm to meters

        # Convert depth to point cloud
        points = depth_to_pointcloud(depth, intrinsic_matrix)

        # Create segmentation mask
        height, width = depth.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Method 1: Use annotations if available
        if filename in annotations:
            # Draw table mask from polygons
            table_polygons = annotations[filename]["table"]
            for polygon in table_polygons:
                points_array = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [points_array], 1)

        # Method 2: Use height threshold if provided
        elif table_height_threshold is not None:
            # Reshape points to image grid
            point_image = np.zeros((height, width, 3))
            valid_mask = depth > 0

            # Fill in valid points
            points_counter = 0
            for h in range(height):
                for w in range(width):
                    if valid_mask[h, w]:
                        point_image[h, w] = points[points_counter]
                        points_counter += 1

            # Create mask based on height (Y-coordinate in camera space)
            table_mask = (point_image[:, :, 1] > table_height_threshold) & valid_mask
            mask[table_mask] = 1

        # Save outputs
        cv2.imwrite(os.path.join(output_dir, "depth", f"{filename}.png"), (depth * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(output_dir, "labels", f"{filename}.png"), mask)

        # Create colored point cloud for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color based on segmentation (flatten mask to match points)
        colors = np.zeros((len(points), 3))
        flat_mask = mask.flatten()[depth.flatten() > 0]
        colors[flat_mask == 1] = [1, 0, 0]  # Table: Red
        colors[flat_mask == 0] = [0, 0, 1]  # Background: Blue
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud(
            os.path.join(output_dir, "point_clouds", f"{filename}.ply"), 
            pcd
        )

    print(f"Created dataset with {len(rgb_files)} samples in {output_dir}")


def augment_point_cloud(points, rotation_range=[-0.1, 0.1], translation_range=[-0.1, 0.1], noise_std=0.01):
    """
    Augment point cloud with random transformations

    Args:
        points: Nx3 array of points
        rotation_range: Range of rotation angles in radians
        translation_range: Range of translations
        noise_std: Standard deviation of Gaussian noise

    Returns:
        augmented_points: Nx3 array of augmented points
    """
    # Random rotation around z-axis (vertical)
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    rot_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # Random translation
    tx = np.random.uniform(translation_range[0], translation_range[1])
    ty = np.random.uniform(translation_range[0], translation_range[1])
    tz = np.random.uniform(translation_range[0], translation_range[1])
    translation = np.array([tx, ty, tz])

    # Apply rotation
    rotated_points = np.dot(points, rot_z.T)

    # Apply translation
    translated_points = rotated_points + translation

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=translated_points.shape)
        noisy_points = translated_points + noise
    else:
        noisy_points = translated_points

    return noisy_points


def preprocess_point_cloud(points, num_points=4096, normalize=True):
    """
    Preprocess point cloud for model input

    Args:
        points: Nx3 array of points
        num_points: Number of points to sample
        normalize: Whether to normalize the point cloud

    Returns:
        processed_points: num_points x 3 array of processed points
    """
    # Sample points if needed
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        sampled_points = points[indices]
    elif len(points) < num_points:
        # Repeat points if we have too few
        indices = np.random.choice(len(points), num_points, replace=True)
        sampled_points = points[indices]
    else:
        sampled_points = points

    # Normalize point cloud
    if normalize:
        center = np.mean(sampled_points, axis=0)
        sampled_points = sampled_points - center

        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(sampled_points**2, axis=1)))
        sampled_points = sampled_points / max_dist

    return sampled_points


def visualize_segmentation(points, labels, output_path=None):
    """
    Visualize point cloud segmentation

    Args:
        points: Nx3 array of points
        labels: N array of labels (0: background, 1: table)
        output_path: Optional path to save visualization
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color based on labels
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [0, 0, 1]  # Background: Blue
    colors[labels == 1] = [1, 0, 0]  # Table: Red
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save point cloud if output path is provided
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)

    # Visualize
    o3d.visualization.draw_geometries([pcd])
