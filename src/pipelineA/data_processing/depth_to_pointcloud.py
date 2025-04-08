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
        tuple: (fx, fy, cx, cy) camera intrinsics parameters, or (None, None, None, None) if parsing fails
    """
    try:
        with open(intrinsics_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()] # Read non-empty lines

        if len(lines) != 3:
            print(f"Error: Expected 3 lines in intrinsics file {intrinsics_path}, found {len(lines)}. Cannot parse.")
            return None, None, None, None # Indicate failure

        # Parse the 3x3 matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        line1_parts = lines[0].split()
        line2_parts = lines[1].split()
        # line3_parts = lines[2].split() # Don't need the third line [0, 0, 1]

        if len(line1_parts) != 3 or len(line2_parts) != 3:
             print(f"Error: Expected 3 values per line in first two lines of intrinsics file {intrinsics_path}. Cannot parse.")
             return None, None, None, None # Indicate failure

        # Check if the matrix format looks correct (0s in expected places)
        if float(line1_parts[1]) != 0 or float(line2_parts[0]) != 0:
             print(f"Warning: Unexpected non-zero values in intrinsics matrix format in {intrinsics_path}.")
             # Proceed anyway, but this might indicate an issue

        fx = float(line1_parts[0])
        cx = float(line1_parts[2])
        fy = float(line2_parts[1])
        cy = float(line2_parts[2])

        return fx, fy, cx, cy
    except Exception as e:
        print(f"Error reading or parsing intrinsics file {intrinsics_path}: {e}")
        return None, None, None, None # Indicate failure


def load_depth_map(depth_path): # Removed use_raw_depth argument
    """Load a depth map from file and scale if necessary.

    Assumes uint16 depth maps are in millimeters and converts them to meters.
    Assumes float32 depth maps (e.g., from .npy) are already in meters.

    Args:
        depth_path (str): Path to the depth map file

    Returns:
        numpy.ndarray: Depth map as a 2D float32 array (in meters)
    """
    # Get file extension
    extension = os.path.splitext(depth_path)[1].lower()
    depth_map = None
    original_dtype = None

    try:
        if extension in ['.png', '.jpg', '.jpeg']:
            # Load image using OpenCV, preserving original bit depth
            img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if img is None:
                raise ValueError("cv2.imread failed")
            original_dtype = img.dtype
            depth_map = img.astype(np.float32) # Convert to float for processing

        elif extension == '.npy':
            # Load numpy array
            depth_map = np.load(str(depth_path))
            original_dtype = depth_map.dtype
            depth_map = depth_map.astype(np.float32) # Ensure float for processing
        else:
            raise ValueError(f"Unsupported depth file extension: {extension}")

        # Scale if the original data was uint16 (assumed millimeters based on analysis)
        if original_dtype == np.uint16:
            # print(f"Debug: Scaling uint16 depth map {depth_path} by 1000.0") # Optional debug
            depth_map /= 1000.0
        elif original_dtype != np.float32:
             # If it's not uint16 or float32, we might not know the scale
             print(f"Warning: Loaded depth map {depth_path} with unexpected dtype {original_dtype}. Assuming meters.")


    except Exception as e:
        print(f"Error loading depth file {depth_path}: {e}")
        # Return a zero array on error to avoid downstream issues immediately
        # Note: This might mask loading errors if not monitored.
        depth_map = np.zeros((480, 640), dtype=np.float32) # Assuming default size

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
        warning_msg = f"Warning: No valid depth values in depth map"
        if depth_path:
            warning_msg += f" ({str(depth_path)})"
        warning_msg += f". Returning empty point cloud."
        print(warning_msg)
        # Return an empty array instead of a dummy cloud
        return np.empty((0, 3), dtype=np.float32)

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
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value

    Returns:
        numpy.ndarray: Point cloud as Nx3 array of (x, y, z) coordinates, or empty array if intrinsics are invalid
    """
    # Read intrinsics
    fx, fy, cx, cy = read_intrinsics(intrinsics_path)

    # Check if intrinsics were read successfully
    if fx is None:
        print(f"Error: Invalid intrinsics for {depth_path}. Cannot create point cloud.")
        return np.empty((0, 3), dtype=np.float32) # Return empty array

    # Load depth map (now always scaled if originally uint16)
    depth_map = load_depth_map(depth_path)

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
        min_depth (float): Minimum valid depth value
        max_depth (float): Maximum valid depth value

    Returns:
        tuple: (points, colors) where points is a Nx3 array and colors is a Nx3 array, or (empty_array, empty_array) if intrinsics are invalid or no valid depth.
    """
    # Read intrinsics
    fx, fy, cx, cy = read_intrinsics(intrinsics_path)

    # Check if intrinsics were read successfully
    if fx is None:
        print(f"Error: Invalid intrinsics for {depth_path}. Cannot create RGBD point cloud.")
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Load depth map (now always scaled if originally uint16)
    depth_map = load_depth_map(depth_path)

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
        warning_msg = f"Warning: No valid depth values in depth map"
        if depth_path:
            warning_msg += f" ({str(depth_path)})"
        warning_msg += f". Returning empty RGBD point cloud."
        print(warning_msg)
        # Return empty arrays instead of a dummy cloud
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

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
