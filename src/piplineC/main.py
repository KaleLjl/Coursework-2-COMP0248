import os
import argparse
import yaml
import torch
import numpy as np
import cv2
from torch.utils.data import random_split
import open3d as o3d

# Import our modules
from dataloader import DepthTableDataset
from model import TableSegmentationModel
from train import Trainer
from evaluate import Evaluator
from utils import SEED

def parse_args():
    parser = argparse.ArgumentParser(description="Table Segmentation Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "inference"],
                        help="Pipeline mode: train, test, or inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--depth_image", type=str, default=None, help="Path to depth image for inference")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load camera intrinsics
    intrinsic_matrix = np.array(config["camera_intrinsics"])
    
    # Create model
    model = TableSegmentationModel(num_classes=2)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    if args.mode == "train":
        # Create dataset
        dataset = DepthTableDataset(
            depth_dir=config["depth_dir"],
            label_dir=config["label_dir"],
            intrinsic_matrix=intrinsic_matrix
        )
        
        # Split dataset
        dataset_size = len(dataset)
        train_size = int(dataset_size * config["train_split"])
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config["seed"])
        )
        
        print(f"Training with {len(train_dataset)} samples and validating with {len(val_dataset)} samples")
        
        # Create trainer
        trainer = Trainer(model, train_dataset, val_dataset, config["training"])
        
        # Train model
        trainer.train()
        
    elif args.mode == "test":
        # Create test dataset
        test_dataset = DepthTableDataset(
            depth_dir=config["test_depth_dir"],
            label_dir=config["test_label_dir"],
            intrinsic_matrix=intrinsic_matrix
        )
        
        print(f"Testing with {len(test_dataset)} samples")
        
        # Create evaluator
        evaluator = Evaluator(model, test_dataset, config["evaluation"])
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Print summary
        print("\nEvaluation Results:")
        print("===================")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
    elif args.mode == "inference":
        # Check if depth image is provided
        if not args.depth_image:
            print("Error: --depth_image is required for inference mode")
            return
        
        # Load depth image
        depth_img = cv2.imread(args.depth_image, cv2.IMREAD_UNCHANGED) / 1000.0  # Convert mm to meters
        
        # Create evaluator for inference
        evaluator = Evaluator(model, None, config["evaluation"])
        
        # Run inference
        pcd, predictions = evaluator.inference(depth_img, intrinsic_matrix)
        
        # Save results
        output_dir = config["evaluation"]["output_dir"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.basename(args.depth_image).split('.')[0]
        output_path = os.path.join(output_dir, f"{filename}_segmented.ply")
        o3d.io.write_point_cloud(output_path, pcd)
        
        # Count table points
        table_points = np.sum(predictions == 1)
        total_points = len(predictions)
        print(f"Table points: {table_points} out of {total_points} ({table_points/total_points*100:.2f}%)")
        
        # Visualize point cloud
        print(f"Saved segmented point cloud to {output_path}")
        print("Visualizing point cloud...")
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
