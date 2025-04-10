import pickle
import numpy as np
import torch
import open3d as o3d
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix

DATA_CONFIG = {
    "data_root": "data/CW2-Dataset/data",
    "num_points": 5120,
    "batch_size": 8,
    "num_workers": 8
}
TRAINGING_CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 200,
    "weight_decay": 1e-5,
    "lr_decay": 0.7,
    "lr_decay_step": 40,
    "early_stop": 20
}
MIT_SEQUENCES = ["mit_32_d507", "mit_76_459", "mit_76_studyroom",
                 "mit_gym_z_squash", "mit_lab_hj"]
HARVARD_SEQUENCES = ["harvard_c5", "harvard_c6", "harvard_c11", "harvard_tea_2"]


def get_mask(polygons, image_shape):
    """
    Generate image-level label based on polygon annotations.

    Args:
        polygons (list): List of dictionaries containing polygon annotations
        image_shape (tuple): Shape of the image (height, width)

    Returns:
        mask: a mask with 1s inside the table polygons
    """
    # Create empty mask
    binary_label = 0
    mask = np.zeros(image_shape, dtype=np.uint8)

    # If no polygons, return 0 label and empty mask
    if not polygons:
        return binary_label, mask

    # Draw all polygons on the mask
    for polygon in polygons:
        points = polygon["points"]
        if len(points) < 3:  # Need at least 3 points to form a polygon
            continue

        points_array = np.array(points, dtype=np.int32)

        # Check if points array is valid (not containing NaNs or out-of-bounds values)
        if np.any(np.isnan(points_array)) or np.any(points_array < 0) or \
           np.any(points_array[:, 0] >= image_shape[1]) or np.any(points_array[:, 1] >= image_shape[0]):
            # Skip invalid polygons
            continue

        cv2.fillPoly(mask, [points_array], 1)
        binary_label = 1  # Set label to 1 if any polygon is drawn

    return binary_label, mask


def get_intrinsics(intrinsics_file):
    """
    Get camera intrinsics from a file.

    Args:
        intrinsics_file (str): Path to the intrinsics file

    Returns:
        dict: Camera intrinsics including fx, fy, cx, cy
    """
    # Check if the file exists
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")

    # Default values in case we can't parse the file
    default_intrinsics = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5
    }

    try:
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()

        if len(lines) >= 3:
            # Parse first row for fx and cx
            row1 = lines[0].strip().split()
            if len(row1) >= 3:
                fx = float(row1[0])
                cx = float(row1[2])
            else:
                print(f"Warning: Could not parse fx and cx from {intrinsics_file}. Using default values.")
                fx, cx = default_intrinsics['fx'], default_intrinsics['cx']

            # Parse second row for fy and cy
            row2 = lines[1].strip().split()
            if len(row2) >= 3:
                fy = float(row2[1])
                cy = float(row2[2])
            else:
                print(f"Warning: Could not parse fy and cy from {intrinsics_file}. Using default values.")
                fy, cy = default_intrinsics['fy'], default_intrinsics['cy']

            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
        else:
            print(f"Warning: Intrinsics file {intrinsics_file} does not have enough lines. Using default values.")
            return default_intrinsics

    except Exception as e:
        print(f"Error reading intrinsics from {intrinsics_file}: {e}. Using default values.")
        return default_intrinsics


def get_polygon(labels_file):
    """
    Read and parse polygon annotations from a file.

    Args:
        labels_file (str): Path to the labels file

    Returns:
        dict: Dictionary mapping image timestamps to polygon annotations
    """
    # Check if the file exists
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        return {}

    annotations = {}

    try:
        # Get the directory where labels_file is located
        labels_dir = os.path.dirname(labels_file)
        possible_depth_dirs = ['depthTSDF', 'depth']

        # Load the table polygon labels
        with open(labels_file, 'rb') as label_file:
            tabletop_labels = pickle.load(label_file)

        # Get list of image files in the same order as the labels
        for possible_dir in possible_depth_dirs:
            depth_dir = os.path.join(os.path.dirname(labels_dir), possible_dir)
            if os.path.exists(depth_dir):
                depth_list = sorted(os.listdir(depth_dir))

                # Map each image to its corresponding polygon labels
                for i, (polygon_list, depth_name) in enumerate(zip(tabletop_labels, depth_list)):
                    # Extract timestamp from image filename
                    timestamp = depth_name[:-4]  # Remove the extension (.png)

                    # Convert polygon format to our internal format
                    # In pickle file, polygons are stored as [frame][table_instance][coordinate]
                    # where coordinate is [x_coords, y_coords]
                    formatted_polygons = []

                    for polygon in polygon_list:
                        # Create a list of (x,y) tuples from the polygon coordinates
                        points = []
                        for x, y in zip(polygon[0], polygon[1]):
                            points.append((float(x), float(y)))

                        if points:  # Only add if there are points
                            formatted_polygons.append({
                                "label": "table",
                                "points": points
                            })

                    annotations[timestamp] = formatted_polygons

        return annotations

    except Exception as e:
        print(f"Error reading pickle annotations from {labels_file}: {e}")


def depth_to_pointcloud(depth_map, intrinsics, subsample=True,
                        num_points=1024, min_depth=0.5, max_depth=10.0):
    """
    Convert depth map to point cloud using camera intrinsics with Open3D.

    Args:
        depth_map (numpy.ndarray): Input depth map in meters
        intrinsics (dict): Camera intrinsics including fx, fy, cx, cy
        subsample (bool): Whether to subsample the point cloud
        num_points (int): Number of points to sample if subsample is True
        min_depth (float): Minimum valid depth value in meters
        max_depth (float): Maximum valid depth value in meters

    Returns:
        numpy.ndarray: Point cloud of shape (N, 3) where N is num_points if
        subsample is True or the number of valid depth pixels otherwise
    """
    # Check if depth map is valid
    if depth_map is None or depth_map.size == 0:
        print("Warning: Empty depth map received, returning zero point cloud")
        return np.zeros((num_points, 3))

    # Create Open3D intrinsic object
    height, width = depth_map.shape

    # Ensure intrinsics are available
    required_keys = ['fx', 'fy', 'cx', 'cy']
    if not all(key in intrinsics for key in required_keys):
        print(f"Warning: Missing intrinsics keys: {[key for key in required_keys if key not in intrinsics]}")
        # Use default values for missing keys
        default_values = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
        for key in required_keys:
            if key not in intrinsics:
                intrinsics[key] = default_values[key]

    # Create Open3D camera intrinsics
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, 
        intrinsics['fx'], intrinsics['fy'], 
        intrinsics['cx'], intrinsics['cy']
    )

    # Create depth image from numpy array
    # Open3D expects depth in meters, so we don't need to convert if already in meters
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))

    # Create point cloud from depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        o3d_intrinsics,
        depth_scale=1.0,  # depth is already in meters
        depth_trunc=max_depth,
        stride=1
    )

    # Get points as numpy array
    points = np.asarray(pcd.points)

    # Filter points based on min depth
    if points.shape[0] > 0:
        # Calculate depth of each point (z coordinate)
        depths = points[:, 2]
        valid_mask = depths >= min_depth
        points = points[valid_mask]

    # Check if we have any valid points
    if points.shape[0] == 0:
        print("Warning: No valid points after filtering, returning zero point cloud")
        return np.zeros((num_points, 3))

    # Subsample the point cloud if needed
    if subsample:
        if points.shape[0] > num_points:
            # Randomly sample points
            indices = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < num_points:
            # Pad with duplicated points or zeros if not enough points
            if points.shape[0] > 0:
                # Duplicate some points
                padding_indices = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
                padding = points[padding_indices]
            else:
                # Use zeros if no valid points
                padding = np.zeros((num_points - points.shape[0], 3))
            points = np.vstack((points, padding))

    return points.astype(np.float32)


def compute_metrics(y_true, y_pred, num_classes=2):
    """
    Compute metrics for the pipline C.

    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        num_classes (int): Number of classes

    Returns:
        dict: Dictionary containing segmentation metrics
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Flatten the arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Check if we have data for both classes
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_labels = np.unique(np.concatenate([unique_true, unique_pred]))

    # Compute per-class metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Compute precision, recall, and F1 with different averaging methods
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Compute per-class metrics (for binary segmentation)
    if num_classes == 2:
        # Get per-class scores (safely handling cases where only one class is present)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Initialize with default values
        precision_table = 0.0
        recall_table = 0.0
        f1_table = 0.0

        # Check if class 1 (table) is present in the scores
        if 1 in unique_labels and len(per_class_precision) > 1:
            precision_table = per_class_precision[1]
            recall_table = per_class_recall[1]
            f1_table = per_class_f1[1]
    else:
        precision_table = np.nan
        recall_table = np.nan
        f1_table = np.nan

    # Compute IoU for each class
    # Always create confusion matrix with all classes (0 to num_classes-1)
    # This ensures we have the expected dimensions even when only one class is present
    conf_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    # Class names for better readability
    class_names = ["background", "table"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]

    iou_list = []
    for cls in range(num_classes):
        intersection = conf_matrix[cls, cls]
        union = np.sum(conf_matrix[cls, :]) + np.sum(conf_matrix[:, cls]) - intersection
        iou = intersection / union if union > 0 else 0.0
        iou_list.append(iou)

    mean_iou = np.mean(iou_list)

    # Create metrics dictionary with standardized naming
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_table': precision_table,
        'recall_table': recall_table,
        'f1_table': f1_table,
        'mean_iou': mean_iou,
    }

    # Add class-specific IoU values with descriptive names
    for cls in range(num_classes):
        metrics[f'iou_{class_names[cls]}'] = iou_list[cls]

    return metrics
