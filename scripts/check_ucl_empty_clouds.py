import os
import glob
import numpy as np
from pathlib import Path
import sys

# Add project root and src directory to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

try:
    # Import necessary functions and configurations
    from pipelineA.config import UCL_DATA_CONFIG, POINT_CLOUD_PARAMS
    from pipelineA.data_processing.depth_to_pointcloud import create_pointcloud_from_depth
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory or the environment is set up correctly.")
    sys.exit(1)

def check_ucl_empty_clouds():
    """
    Checks the UCL dataset for frames that produce empty point clouds
    using the current point cloud generation logic.
    """
    print("Starting UCL Empty Point Cloud Check...")

    # Get config values
    try:
        ucl_base_path = Path(UCL_DATA_CONFIG['base_path'])
        min_depth = POINT_CLOUD_PARAMS['min_depth']
        max_depth = POINT_CLOUD_PARAMS['max_depth']
    except KeyError as e:
        print(f"Error: Missing key in configuration: {e}")
        return

    depth_dir = ucl_base_path / "depth"
    intrinsics_path = ucl_base_path / "intrinsics.txt"

    if not ucl_base_path.exists():
        print(f"Error: UCL base path not found: {ucl_base_path}")
        return
    if not depth_dir.exists():
        print(f"Error: UCL depth directory not found: {depth_dir}")
        return
    if not intrinsics_path.exists():
        print(f"Error: UCL intrinsics file not found: {intrinsics_path}")
        # Note: create_pointcloud_from_depth also checks this, but good to check early
        # return # Allow the main function to handle this per-file

    # Find depth files (assuming .png based on previous analysis)
    depth_files = sorted(glob.glob(str(depth_dir / "*.png")))

    if not depth_files:
        print(f"Warning: No depth files (.png) found in {depth_dir}")
        return

    total_files = len(depth_files)
    empty_count = 0
    error_count = 0 # Count errors during point cloud creation

    print(f"Found {total_files} depth files in {depth_dir}.")
    print(f"Using Intrinsics: {intrinsics_path}")
    print(f"Using min_depth={min_depth}, max_depth={max_depth}")

    for i, depth_file_path in enumerate(depth_files):
        print(f"Processing file {i+1}/{total_files}: {Path(depth_file_path).name}...", end='\r')

        try:
            # Generate point cloud using the corrected function
            # Note: use_raw_depth argument is no longer needed in the updated function
            points = create_pointcloud_from_depth(
                depth_path=str(depth_file_path),
                intrinsics_path=str(intrinsics_path),
                min_depth=min_depth,
                max_depth=max_depth
            )

            # Check if the resulting point cloud is empty
            if points.shape[0] == 0:
                empty_count += 1
                # Optionally print which file resulted in empty cloud
                # print(f"\n  -> Empty point cloud for: {Path(depth_file_path).name}")

        except Exception as e:
            print(f"\nError processing {Path(depth_file_path).name}: {e}")
            error_count += 1
            # Continue to next file

    print("\nCheck Complete.")
    print("--------------------")
    print(f"Total UCL depth files checked: {total_files}")
    print(f"Files resulting in empty point clouds: {empty_count}")
    if total_files > 0:
        percentage_empty = (empty_count / total_files) * 100
        print(f"Percentage empty: {percentage_empty:.2f}%")
    if error_count > 0:
         print(f"Errors encountered during processing: {error_count}")
    print("--------------------")

if __name__ == "__main__":
    # Ensure the script is run from the project root for correct path resolution
    os.chdir(PROJECT_ROOT)
    check_ucl_empty_clouds()
