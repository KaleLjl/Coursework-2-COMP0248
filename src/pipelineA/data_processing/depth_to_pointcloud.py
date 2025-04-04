import numpy as np
import cv2
import os
from pathlib import Path
import open3d as o3d

def read_intrinsics(intrinsics_path):
    """Read camera intrinsics from a file.
    
    Args:
        intrinsics_path (str): Path to the intrinsics file
        
    Returns:
        tuple: (fx, fy, cx, cy) camera intrinsics parameters
    """
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the intrinsics matrix
    line1 = lines[0].strip().split()
    line2 = lines[1].strip().split()
    
    fx = float(line1[0])
    fy = float(line2[1])
    cx = float(line1[2])
    cy = float(line2[2])
    
    return fx, fy, cx, cy

def load_depth_map(depth_path, use_raw_depth=False):
    """Load a depth map from file.
    
    Args:
        depth_path (str): Path to the depth map file
        use_raw_depth (bool): If True, use raw depth, otherwise use depthTSDF
        
    Returns:
        numpy.ndarray: Depth map as a 2D array
    """
    # Get file extension
    extension = os.path.splitext(depth_path)[1].lower()
    
    # For raw depth maps (harvard_tea_2)
    if use_raw_depth:
        if extension in ['.png', '.jpg', '.jpeg']:
            depth_map_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_map_raw is None:
                 print(f"Error: Failed to read raw depth image {depth_path}")
                 depth_map = np.zeros((480, 640), dtype=np.float32)
            else:
                 # Removed the debug print statement
                 # Usually raw depth is in millimeters, convert to meters
                 depth_map = depth_map_raw.astype(np.float32) / 1000.0
        else: # Corrected indentation level
            # Try to load as numpy array
            try:
                depth_map = np.load(depth_path)
            except Exception as e:
                print(f"Error loading raw depth file {depth_path}: {e}")
                depth_map = np.zeros((480, 640), dtype=np.float32)
    # For TSDF processed depth maps
    else:
        if extension == '.npy':
            depth_map = np.load(depth_path)
        elif extension == '.png':
            # Some datasets might store depth as PNG
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth_map = depth_map.astype(np.float32) / 1000.0
        else:
            # Try different approaches
            try:
                # Try to load as numpy binary
                depth_map = np.load(depth_path)
            except Exception:
                try:
                    # Try to load as image
                    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                    depth_map = depth_map.astype(np.float32) / 1000.0
                except Exception as e:
                    print(f"Error loading depth file {depth_path}: {e}")
                    depth_map = np.zeros((480, 640), dtype=np.float32)
    
    return depth_map

def depth_to_pointcloud(depth_map, fx, fy, cx, cy, min_depth=0.1, max_depth=10.0, depth_path=None):
    """Convert depth map to point cloud.
    
    Args:
        depth_map (numpy.ndarray): Input depth map
        fx (float): Focal length in x direction
        fy (float): Focal length in y direction
        cx (float): Principal point x coordinate
        cy (float): Principal point y coordinate
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
        
    Returns:
        numpy.ndarray: Point cloud as Nx3 array of (x, y, z) coordinates
    """
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create pixel coordinate grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Filter out invalid depth values
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    # Check if we have any valid depth values
    if np.sum(valid_mask) == 0:
        # Return a small dummy point cloud instead of an empty one
        # Create a 3x3x3 grid of points centered at origin
        dummy_size = 3
        x = np.linspace(-1, 1, dummy_size)
        y = np.linspace(-1, 1, dummy_size)
        z = np.linspace(-1, 1, dummy_size)
        xv, yv, zv = np.meshgrid(x, y, z)
        points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
        warning_msg = f"Warning: No valid depth values in depth map"
        if depth_path:
            # Print the full path instead of just the basename
            warning_msg += f" ({str(depth_path)})" 
        warning_msg += f". Created dummy point cloud with {points.shape[0]} points."
        print(warning_msg)
        return points
    
    # Get valid coordinates and depths
    valid_u = u[valid_mask]
    valid_v = v[valid_mask]
    valid_depth = depth_map[valid_mask]
    
    # Calculate 3D coordinates
    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth
    
    # Stack coordinates into 3D points
    points = np.stack([x, y, z], axis=1)
    
    return points

def create_pointcloud_from_depth(depth_path, intrinsics_path, use_raw_depth=False, 
                                min_depth=0.1, max_depth=10.0):
    """Create point cloud from depth map using camera intrinsics.
    
    Args:
        depth_path (str): Path to the depth map file
        intrinsics_path (str): Path to the intrinsics file
        use_raw_depth (bool): If True, use raw depth, otherwise use depthTSDF
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
        
    Returns:
        numpy.ndarray: Point cloud as Nx3 array of (x, y, z) coordinates
    """
    # Read intrinsics
    fx, fy, cx, cy = read_intrinsics(intrinsics_path)
    
    # Load depth map
    depth_map = load_depth_map(depth_path, use_raw_depth)
    
    # Convert to point cloud, passing the path for logging
    points = depth_to_pointcloud(depth_map, fx, fy, cx, cy, min_depth, max_depth, depth_path=depth_path)
    
    return points

def visualize_pointcloud(points, colors=None):
    """Visualize a point cloud using Open3D.
    
    Args:
        points (numpy.ndarray): Nx3 array of (x, y, z) coordinates
        colors (numpy.ndarray, optional): Nx3 array of RGB colors
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def create_rgbd_pointcloud(depth_path, image_path, intrinsics_path, use_raw_depth=False, 
                          min_depth=0.1, max_depth=10.0):
    """Create colored point cloud from depth map and RGB image.
    
    Args:
        depth_path (str): Path to the depth map file
        image_path (str): Path to the RGB image file
        intrinsics_path (str): Path to the intrinsics file
        use_raw_depth (bool): If True, use raw depth, otherwise use depthTSDF
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value
        
    Returns:
        tuple: (points, colors) where points is a Nx3 array and colors is a Nx3 array
    """
    # Read intrinsics
    fx, fy, cx, cy = read_intrinsics(intrinsics_path)
    
    # Load depth map
    depth_map = load_depth_map(depth_path, use_raw_depth)
    
    # Load RGB image
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Ensure image and depth have the same dimensions
    if rgb_image.shape[:2] != depth_map.shape:
        rgb_image = cv2.resize(rgb_image, (depth_map.shape[1], depth_map.shape[0]))
    
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create pixel coordinate grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Filter out invalid depth values
    valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
    
    # Check if we have any valid depth values
    if np.sum(valid_mask) == 0:
        # Return a small dummy point cloud instead of an empty one
        # Create a 3x3x3 grid of points centered at origin
        dummy_size = 3
        x = np.linspace(-1, 1, dummy_size)
        y = np.linspace(-1, 1, dummy_size)
        z = np.linspace(-1, 1, dummy_size)
        xv, yv, zv = np.meshgrid(x, y, z)
        points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
        # Create random colors for the dummy points
        colors = np.random.random((points.shape[0], 3))
        # Include full path in the warning
        print(f"Warning: No valid depth values in depth map ({str(depth_path)}). Created dummy point cloud with {points.shape[0]} points.")
        return points, colors
    
    # Get valid coordinates and depths
    valid_u = u[valid_mask]
    valid_v = v[valid_mask]
    valid_depth = depth_map[valid_mask]
    
    # Calculate 3D coordinates
    x = (valid_u - cx) * valid_depth / fx
    y = (valid_v - cy) * valid_depth / fy
    z = valid_depth
    
    # Stack coordinates into 3D points
    points = np.stack([x, y, z], axis=1)
    
    # Get the colors for valid points (normalize to 0-1)
    colors = rgb_image[valid_mask] / 255.0
    
    return points, colors
