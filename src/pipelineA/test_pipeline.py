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
    TRAIN_SEQUENCES, TEST1_SEQUENCES,
    POINT_CLOUD_PARAMS, MODEL_PARAMS
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

# Use absolute path for testing
BASE_DATA_DIR = Path("/cs/student/projects1/rai/2024/jialeli/Objection-Coursework2/data/CW2-Dataset/data")

def test_depth_to_pointcloud():
    """Test depth to pointcloud conversion."""
    print("\n=== Testing Depth to Pointcloud Conversion ===")
    
    # Choose a sample from the dataset - use harvard_c5 which we know exists
    sequence_name = "harvard_c5"
    sub_sequence = TEST1_SEQUENCES[sequence_name][0]
    
    print(f"Using sequence: {sequence_name}/{sub_sequence}")
    
    # Get depth file and intrinsics
    sequence_dir = BASE_DATA_DIR / sequence_name / sub_sequence
    depth_dir = sequence_dir / "depthTSDF"
    depth_files = list(depth_dir.glob("*.png"))
    if not depth_files:
        print("No depth files found!")
        return False
    
    depth_file = str(depth_files[0])
    intrinsics_path = str(sequence_dir / "intrinsics.txt")
    image_dir = sequence_dir / "image"
    image_files = list(image_dir.glob("*.jpg"))  # Changed to .jpg
    image_file = str(image_files[0]) if image_files else None
    
    print(f"Depth file: {depth_file}")
    print(f"Intrinsics: {intrinsics_path}")
    print(f"Image file: {image_file}")
    
    # Convert to pointcloud
    try:
        points = create_pointcloud_from_depth(
            depth_file, intrinsics_path, use_raw_depth=False,
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
            points, colors = create_rgbd_pointcloud(
                depth_file, image_file, intrinsics_path, use_raw_depth=False,
                min_depth=POINT_CLOUD_PARAMS['min_depth'],
                max_depth=POINT_CLOUD_PARAMS['max_depth']
            )
            print(f"Colored point cloud shape: {points.shape}, colors shape: {colors.shape}")
        
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
    
    try:
        # Create dataset with a small subset for testing - use harvard_c5
        test_sequences = {"harvard_c5": TEST1_SEQUENCES["harvard_c5"]}
        
        dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            sequences=test_sequences,
            augment=False,
            mode='test',
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