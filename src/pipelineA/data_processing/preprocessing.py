import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def normalize_point_cloud(points):
    """Normalize point cloud to have zero mean and unit sphere bounding.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        
    Returns:
        numpy.ndarray: Normalized point cloud
    """
    # Check if we have enough points
    if points.shape[0] < 2:
        print(f"Warning: Not enough points in point cloud (only {points.shape[0]}). Returning dummy normalized points.")
        # Return a small dummy point cloud
        dummy_size = 3
        x = np.linspace(-0.5, 0.5, dummy_size)
        y = np.linspace(-0.5, 0.5, dummy_size)
        z = np.linspace(-0.5, 0.5, dummy_size)
        xv, yv, zv = np.meshgrid(x, y, z)
        normalized_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
        return normalized_points
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Center the point cloud
    centered_points = points - centroid
    
    # Calculate the maximum distance from origin
    max_distance = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
    
    # Scale to unit sphere
    if max_distance > 1e-6:  # Avoid division by near-zero values
        normalized_points = centered_points / max_distance
    else:
        print("Warning: Point cloud has all points at the same location or very close. Cannot normalize properly.")
        # Return a small dummy point cloud
        dummy_size = 3
        x = np.linspace(-0.5, 0.5, dummy_size)
        y = np.linspace(-0.5, 0.5, dummy_size)
        z = np.linspace(-0.5, 0.5, dummy_size)
        xv, yv, zv = np.meshgrid(x, y, z)
        normalized_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
        
    return normalized_points

def random_sample_points(points, colors=None, num_points=2048):
    """Randomly sample points from a point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        num_points (int): Number of points to sample
        
    Returns:
        tuple: (sampled_points, sampled_colors) where sampled_points is a
            numpy.ndarray of shape (num_points, 3) and sampled_colors is a
            numpy.ndarray of shape (num_points, 3)
    """
    num_input_points = points.shape[0]
    
    # If there are not enough points, duplicate some points
    if num_input_points < num_points:
        # Indices to duplicate (with replacement)
        idxs = np.random.choice(num_input_points, num_points - num_input_points)
        dup_points = points[idxs]
        sampled_points = np.vstack([points, dup_points])
        
        if colors is not None:
            dup_colors = colors[idxs]
            sampled_colors = np.vstack([colors, dup_colors])
        else:
            sampled_colors = None
    # If there are too many points, randomly sample
    elif num_input_points > num_points:
        # Indices to keep (without replacement)
        idxs = np.random.choice(num_input_points, num_points, replace=False)
        sampled_points = points[idxs]
        
        if colors is not None:
            sampled_colors = colors[idxs]
        else:
            sampled_colors = None
    # If there are exactly the right number of points
    else:
        sampled_points = points
        sampled_colors = colors
        
    return sampled_points, sampled_colors

def farthest_point_sampling(points, colors=None, num_points=2048):
    """Sample points using farthest point sampling (FPS).
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        num_points (int): Number of points to sample
        
    Returns:
        tuple: (sampled_points, sampled_colors) where sampled_points is a
            numpy.ndarray of shape (num_points, 3) and sampled_colors is a
            numpy.ndarray of shape (num_points, 3)
    """
    num_input_points = points.shape[0]
    
    # If there are not enough points, use random sampling with duplication
    if num_input_points < num_points:
        return random_sample_points(points, colors, num_points)
    
    # If number of points is exact, return as is
    if num_input_points == num_points:
        return points, colors
    
    # Farthest point sampling
    sampled_indices = np.zeros(num_points, dtype=np.int32)
    # Initialize a random point as the first point
    sampled_indices[0] = np.random.randint(0, num_input_points)
    
    # Calculate distance to the rest of the points
    distances = np.full(num_input_points, np.inf)
    
    # Iteratively select farthest points
    for i in range(1, num_points):
        last_idx = sampled_indices[i-1]
        last_point = points[last_idx]
        
        # Calculate squared distances to the last sampled point
        new_distances = np.sum((points - last_point)**2, axis=1)
        
        # Update distances to be minimum between current and new distances
        distances = np.minimum(distances, new_distances)
        
        # Choose the point with maximum distance
        sampled_indices[i] = np.argmax(distances)
    
    # Get the sampled points
    sampled_points = points[sampled_indices]
    
    # Get the sampled colors if provided
    if colors is not None:
        sampled_colors = colors[sampled_indices]
    else:
        sampled_colors = None
        
    return sampled_points, sampled_colors

def sample_points(points, colors=None, num_points=2048, method='fps'):
    """Sample points from a point cloud using the specified method.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        num_points (int): Number of points to sample
        method (str): Sampling method: 'random', 'fps' (farthest point sampling),
            or None (no sampling)
        
    Returns:
        tuple: (sampled_points, sampled_colors) where sampled_points is a
            numpy.ndarray of shape (num_points, 3) and sampled_colors is a
            numpy.ndarray of shape (num_points, 3)
    """
    if method is None:
        return points, colors
    elif method == 'random':
        return random_sample_points(points, colors, num_points)
    elif method == 'fps':
        return farthest_point_sampling(points, colors, num_points)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

def rotate_point_cloud_y(points, angle_degree):
    """Rotate point cloud around Y axis.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        angle_degree (float): Rotation angle in degrees
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    angle_rad = np.radians(angle_degree)
    cosval = np.cos(angle_rad)
    sinval = np.sin(angle_rad)
    
    # Rotation matrix around Y axis
    rotation_matrix = np.array([
        [cosval, 0, sinval],
        [0, 1, 0],
        [-sinval, 0, cosval]
    ])
    
    # Apply rotation
    rotated_points = np.dot(points, rotation_matrix.T)
    
    return rotated_points

def rotate_point_cloud_z(points, angle_degree):
    """Rotate point cloud around Z axis.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        angle_degree (float): Rotation angle in degrees
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    angle_rad = np.radians(angle_degree)
    cosval = np.cos(angle_rad)
    sinval = np.sin(angle_rad)
    
    # Rotation matrix around Z axis
    rotation_matrix = np.array([
        [cosval, -sinval, 0],
        [sinval, cosval, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    rotated_points = np.dot(points, rotation_matrix.T)
    
    return rotated_points

def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """Add random jitter to point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        sigma (float): Standard deviation of Gaussian noise
        clip (float): Maximum absolute value of jitter
        
    Returns:
        numpy.ndarray: Jittered point cloud
    """
    # Generate random noise
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    
    # Add noise to points
    jittered_points = points + noise
    
    return jittered_points

def scale_point_cloud(points, scale):
    """Scale point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        scale (float): Scale factor
        
    Returns:
        numpy.ndarray: Scaled point cloud
    """
    return points * scale

def point_dropout(points, colors=None, dropout_ratio=0.1):
    """Randomly drop points from the point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        dropout_ratio (float): Ratio of points to drop (0.0 to 1.0)
        
    Returns:
        tuple: (dropped_points, dropped_colors)
    """
    if dropout_ratio <= 0.0:
        return points, colors
        
    num_points = points.shape[0]
    num_keep = int(num_points * (1.0 - dropout_ratio))
    
    if num_keep <= 0: # Avoid dropping all points
        print(f"Warning: Point dropout ratio {dropout_ratio} too high, keeping 1 point.")
        num_keep = 1
        
    keep_indices = np.random.choice(num_points, num_keep, replace=False)
    
    dropped_points = points[keep_indices]
    dropped_colors = colors[keep_indices] if colors is not None else None
    
    return dropped_points, dropped_colors

def random_subsample(points, colors=None, subsample_ratio=0.8):
    """Randomly subsample points from the point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        subsample_ratio (float): Ratio of points to keep (0.0 to 1.0)
        
    Returns:
        tuple: (subsampled_points, subsampled_colors)
    """
    if subsample_ratio >= 1.0:
        return points, colors
        
    num_points = points.shape[0]
    num_keep = int(num_points * subsample_ratio)
    
    if num_keep <= 0: # Avoid keeping zero points
        print(f"Warning: Subsample ratio {subsample_ratio} too low, keeping 1 point.")
        num_keep = 1
        
    keep_indices = np.random.choice(num_points, num_keep, replace=False)
    
    subsampled_points = points[keep_indices]
    subsampled_colors = colors[keep_indices] if colors is not None else None
    
    return subsampled_points, subsampled_colors

def augment_point_cloud(points, colors=None, rotation_y_range=None, 
                        rotation_z=False, jitter_sigma=None, jitter_clip=None,
                        scale_range=None, point_dropout_ratio=None,
                        random_subsample_flag=False, subsample_range=None):
    """Apply data augmentation to point cloud.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        rotation_y_range (list, optional): Range of rotation angles around Y axis [min, max]
        rotation_z (bool, optional): Whether to apply random rotation around Z axis
        jitter_sigma (float, optional): Standard deviation for point jittering
        jitter_clip (float, optional): Maximum absolute jitter
        scale_range (list, optional): Range of scaling factors [min, max]
        
    Returns:
        tuple: (augmented_points, colors) where augmented_points is a
            numpy.ndarray of shape (N, 3) and colors is unchanged
    """
    augmented_points = points.copy()
    
    # Apply random rotation around Y axis
    if rotation_y_range is not None:
        angle_y = np.random.uniform(rotation_y_range[0], rotation_y_range[1])
        augmented_points = rotate_point_cloud_y(augmented_points, angle_y)
    
    # Apply random rotation around Z axis
    if rotation_z:
        angle_z = np.random.uniform(0, 360)
        augmented_points = rotate_point_cloud_z(augmented_points, angle_z)
    
    # Apply random jitter
    if jitter_sigma is not None:
        clip = jitter_clip if jitter_clip is not None else jitter_sigma * 5
        augmented_points = jitter_point_cloud(augmented_points, jitter_sigma, clip)
    
    # Apply random scaling
    if scale_range is not None:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented_points = scale_point_cloud(augmented_points, scale)
        
    # Apply random subsampling
    if random_subsample_flag and subsample_range is not None:
        subsample_ratio = np.random.uniform(subsample_range[0], subsample_range[1])
        augmented_points, colors = random_subsample(augmented_points, colors, subsample_ratio)
        
    # Apply point dropout
    if point_dropout_ratio is not None and point_dropout_ratio > 0:
        augmented_points, colors = point_dropout(augmented_points, colors, point_dropout_ratio)
    
    return augmented_points, colors

def preprocess_point_cloud(points, colors=None, normalize=True, num_points=2048,
                          sampling_method='fps', augment=False, augmentation_params=None):
    """Preprocess point cloud with normalization, sampling, and augmentation.
    
    Args:
        points (numpy.ndarray): Input point cloud (Nx3)
        colors (numpy.ndarray, optional): Input colors (Nx3)
        normalize (bool): Whether to normalize the point cloud
        num_points (int): Number of points to sample
        sampling_method (str): Sampling method: 'random', 'fps', or None
        augment (bool): Whether to apply data augmentation
        augmentation_params (dict, optional): Parameters for augmentation
        
    Returns:
        tuple: (processed_points, processed_colors) where processed_points is a
            numpy.ndarray of shape (num_points, 3) and processed_colors is a
            numpy.ndarray of shape (num_points, 3) or None
    """
    # Normalize point cloud
    if normalize:
        points = normalize_point_cloud(points)
    
    # Apply data augmentation if requested and in training mode
    if augment and augmentation_params is not None:
        points, colors = augment_point_cloud(
            points, colors,
            rotation_y_range=augmentation_params.get('rotation_y_range'),
            rotation_z=augmentation_params.get('rotation_z', False),
            jitter_sigma=augmentation_params.get('jitter_sigma'),
            jitter_clip=augmentation_params.get('jitter_clip'),
            scale_range=augmentation_params.get('scale_range'),
            point_dropout_ratio=augmentation_params.get('point_dropout_ratio'),
            random_subsample_flag=augmentation_params.get('random_subsample', False),
            subsample_range=augmentation_params.get('subsample_range')
        )
    
    # Sample points (ensure final count is num_points AFTER augmentation)
    if sampling_method is not None:
        points, colors = sample_points(points, colors, num_points, sampling_method)
    
    return points, colors
