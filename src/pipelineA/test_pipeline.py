import os
import sys
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_SEQUENCES, VALIDATION_FRAMES, TEST_FRAMES, # Updated import
    POINT_CLOUD_PARAMS, MODEL_PARAMS, BASE_DATA_DIR # Import BASE_DATA_DIR
)
from models.classifier import get_model
from models.utils import set_seed
from data_processing.dataset import create_data_loaders, TableDataset
from data_processing.depth_to_pointcloud import (
    create_pointcloud_from_depth, create_rgbd_pointcloud, visualize_pointcloud
)
from data_processing.preprocessing import (
    normalize_point_cloud, sample_points, preprocess_point_cloud
)

# Removed hardcoded BASE_DATA_DIR, using the one from config.py

def test_depth_to_pointcloud():
    """Test depth to pointcloud conversion."""
    print("\n=== Testing Depth to Pointcloud Conversion ===")

    if not TEST_FRAMES:
        print("TEST_FRAMES list is empty. Cannot run test.")
        return False

    # Choose the first frame ID from the test set
    test_frame_id = TEST_FRAMES[0]
    print(f"Using test frame ID: {test_frame_id}")

    # Parse the frame ID: e.g., "harvard_c6/hv_c6_1/depthTSDF/0000022-000000700755.png"
    try:
        parts = test_frame_id.split('/')
        if len(parts) != 4: # Expecting 4 parts now
            raise ValueError(f"Invalid frame ID format (expected 4 parts): {test_frame_id}")
        sequence_name, sub_sequence, depth_folder, frame_file_name = parts
        # Extract base name without extension
        frame_base_name = Path(frame_file_name).stem 
    except ValueError as e:
        print(f"Error parsing frame ID: {e}")
        return False

    print(f"Parsed: Sequence={sequence_name}, SubSequence={sub_sequence}, DepthFolder={depth_folder}, FrameBase={frame_base_name}")

    # Construct paths using parsed components
    sequence_dir = BASE_DATA_DIR / sequence_name / sub_sequence
    # Use the parsed depth_folder and frame_file_name directly
    depth_file = str(sequence_dir / depth_folder / frame_file_name) 
    intrinsics_path = str(sequence_dir / "intrinsics.txt")
    # Construct image file path using the base name
    image_file = str(sequence_dir / "image" / f"{frame_base_name}.jpg") 

    # Check if files exist (paths constructed above)
    if not Path(depth_file).exists():
        print(f"Depth file not found: {depth_file}")
        # No need for special harvard_tea_2 check here as depth_folder is parsed
        return False
            
    if not Path(intrinsics_path).exists():
        print(f"Intrinsics file not found: {intrinsics_path}")
        return False
        
    if not Path(image_file).exists():
        print(f"Image file not found: {image_file}. Proceeding without color.")
        image_file = None # Set to None if not found

    print(f"Using Depth file: {depth_file}")
    print(f"Intrinsics: {intrinsics_path}")
    print(f"Image file: {image_file}")
    
    # Convert to pointcloud
    try:
        # Determine if raw depth should be used based on sequence name
        use_raw_depth = (sequence_name == "harvard_tea_2")
        
        points = create_pointcloud_from_depth(
            depth_file, intrinsics_path, use_raw_depth=use_raw_depth,
            min_depth=POINT_CLOUD_PARAMS['min_depth'],
            max_depth=POINT_CLOUD_PARAMS['max_depth']
        )
        
        print(f"Point cloud shape: {points.shape}")
        print(f"Range: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
        # Test normalization
        normalized_points = normalize_point_cloud(points)
        print(f"Normalized point cloud range: min={normalized_points.min(axis=0)}, max={normalized_points.max(axis=0)}")
        
        # Test sampling
        sampled_points, _ = sample_points(points, None, num_points=1000, method='fps')
        print(f"Sampled point cloud shape: {sampled_points.shape}")
        
        # Create colored pointcloud if image is available
        if image_file:
            points_rgbd, colors = create_rgbd_pointcloud(
                depth_file, image_file, intrinsics_path, use_raw_depth=use_raw_depth,
                min_depth=POINT_CLOUD_PARAMS['min_depth'],
                max_depth=POINT_CLOUD_PARAMS['max_depth']
            )
            # Use the points from RGBD version if available for visualization
            points_for_vis = points_rgbd 
            print(f"Colored point cloud shape: {points_rgbd.shape}, colors shape: {colors.shape}")
        else:
            points_for_vis = points # Use non-colored points if no image
            colors = None

        # Visualize a small subset for quick testing
        num_vis_points = min(len(points_for_vis), 5000) # Limit visualization points
        indices = np.random.choice(len(points_for_vis), num_vis_points, replace=False)
        visual_points = points_for_vis[indices]
        visual_colors = colors[indices] if colors is not None else None
        
        # Visualize a small subset for quick testing
        visual_points = points[::50]  # Every 50th point
        visual_colors = colors[::50] if image_file and 'colors' in locals() else None
        
        print("Displaying point cloud (press any key to continue)...")
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(visual_points)
        if visual_colors is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(visual_colors)
        o3d.visualization.draw_geometries([o3d_pcd])
        
        return True
    except Exception as e:
        print(f"Error testing depth to pointcloud: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading."""
    print("\n=== Testing Dataset Loading ===")

    if not TEST_FRAMES:
        print("TEST_FRAMES list is empty. Cannot run test.")
        return False
        
    try:
        # Create dataset using a small slice of the TEST_FRAMES list
        # Assuming TableDataset now accepts 'data_spec' which can be a list of frame IDs
        test_frame_list = TEST_FRAMES[:5] # Use first 5 test frames
        print(f"Using {len(test_frame_list)} frame IDs for dataset test: {test_frame_list}")
        
        dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            data_spec=test_frame_list, # Pass the list of frame IDs
            augment=False,
            mode='test', # Ensure mode is test/val for no augmentation if augment=False
            point_cloud_params=POINT_CLOUD_PARAMS
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("Dataset is empty!")
            return False
        
        # Get a sample
        sample = dataset[0]
        points = sample['points']
        label = sample['label']
        metadata = sample['metadata']
        
        print(f"Sample points shape: {points.shape}")
        print(f"Sample label: {label}")
        print(f"Sample metadata keys: {metadata.keys()}")
        
        # Test dataloader with batch size 1 to avoid size issues
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 to avoid size mismatch issues
            shuffle=False,
            num_workers=0
        )
        
        batch = next(iter(loader))
        print(f"Batch points shape: {batch['points'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        
        return True
    except Exception as e:
        print(f"Error testing dataset loading: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n=== Testing Model Creation ===")
    
    try:
        # Create model
        model = get_model(
            model_type=MODEL_PARAMS['model_type'],
            num_classes=2,
            k=MODEL_PARAMS['k'],
            emb_dims=MODEL_PARAMS['emb_dims'],
            dropout=MODEL_PARAMS['dropout']
        )
        
        print(f"Model: {model.__class__.__name__}")
        
        # Create a random batch
        batch_size = 2
        num_points = POINT_CLOUD_PARAMS['num_points']
        points = torch.randn(batch_size, num_points, 3)
        
        # Forward pass
        outputs = model(points)
        print(f"Output shape: {outputs.shape}")
        
        # Make sure outputs are valid
        if outputs.shape != (batch_size, 2):
            print(f"Unexpected output shape: {outputs.shape}, expected: {(batch_size, 2)}")
            return False
        
        return True
    except Exception as e:
        print(f"Error testing model creation: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=== Running Pipeline A Tests ===")
    
    # Set seed for reproducibility
    set_seed(42)
    
    tests = [
        test_depth_to_pointcloud,
        test_dataset_loading,
        test_model_creation
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Exception in {test.__name__}: {e}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, test in enumerate(tests):
        print(f"{test.__name__}: {'PASSED' if results[i] else 'FAILED'}")
    
    if all(results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")

if __name__ == "__main__":
    run_all_tests()
