import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to sys.path to allow importing modules from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.pipelineA.data_processing.dataset import TableDataset, collate_fn, create_data_loaders
from src.pipelineA.models.classifier import DGCNN, PointNet # Import model classes
from src.pipelineA.config import (
    BASE_DATA_DIR, TEST_FRAMES, MODEL_PARAMS, POINT_CLOUD_PARAMS
)

def visualize_predictions(args):
    """Loads a model, runs inference on the test set, and saves annotated images."""

    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # --- Load Model ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)

    # Determine model type from checkpoint or args (prefer checkpoint if available)
    # Use MODEL_PARAMS from config as a fallback if not in checkpoint
    model_config = checkpoint.get('config', {}).get('MODEL_PARAMS', MODEL_PARAMS)
    model_type = model_config.get('model_type', MODEL_PARAMS['model_type']) # Default to config if needed

    print(f"Loading model type: {model_type}")
    print(f"Model config used: {model_config}")

    # Instantiate the correct model based on type
    # Ensure num_classes is present, default to 2 if missing
    if 'num_classes' not in model_config:
        print("Warning: 'num_classes' not found in model config, defaulting to 2.")
        model_config['num_classes'] = 2

    if model_type == 'dgcnn':
        # Filter config keys to only those expected by DGCNN constructor
        dgcnn_keys = {'num_classes', 'k', 'emb_dims', 'dropout', 'feature_dropout'}
        filtered_config = {k: v for k, v in model_config.items() if k in dgcnn_keys}
        model = DGCNN(**filtered_config).to(device) # Unpack the filtered dictionary
    elif model_type == 'pointnet':
         # Filter config keys for PointNet
        pointnet_keys = {'num_classes', 'dropout'}
        filtered_config = {k: v for k, v in model_config.items() if k in pointnet_keys}
        model = PointNet(**filtered_config).to(device) # Unpack the filtered dictionary
    else:
        print(f"Error: Unknown model type '{model_type}'")
        return

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_path}")
    except KeyError:
        print(f"Error: 'model_state_dict' not found in checkpoint {model_path}")
        return
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Ensure model architecture in config matches the checkpoint.")
        return

    model.eval() # Set model to evaluation mode

    # --- Load Test Data ---
    # Use create_data_loaders to get the test loader configured correctly
    _, _, test_loader = create_data_loaders(
        data_root=args.data_root,
        test_spec=TEST_FRAMES, # Use the test frame list from config
        batch_size=1, # Process one image at a time
        num_workers=args.num_workers,
        point_cloud_params=POINT_CLOUD_PARAMS # Use point cloud params from config
    )

    if not test_loader:
        print("Error: Failed to create test data loader.")
        return

    print(f"Loaded {len(test_loader.dataset)} test samples.")

    # --- Run Inference and Visualize ---
    class_names = {0: "No Table", 1: "Table"}
    correct_color = (0, 255, 0) # Green
    incorrect_color = (0, 0, 255) # Red

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            points = batch['points'].to(device)
            labels = batch['label'].to(device)
            metadata = batch['metadata'][0] # Batch size is 1

            # Perform inference
            outputs = model(points)
            preds = torch.argmax(outputs, dim=1)

            prediction = preds.item()
            ground_truth = labels.item()
            is_correct = (prediction == ground_truth)

            # Load RGB image
            image_path = metadata.get('image_file')
            if image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                     print(f"Warning: Failed to load image {image_path}")
                     continue

                # Annotate image
                pred_text = f"Predicted: {class_names[prediction]}"
                gt_text = f"Ground Truth: {class_names[ground_truth]}"
                text_color = correct_color if is_correct else incorrect_color

                # Put text on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                y0, dy = 30, 30 # Starting y position and line spacing

                cv2.putText(image, pred_text, (10, y0), font, font_scale, text_color, thickness, cv2.LINE_AA)
                cv2.putText(image, gt_text, (10, y0 + dy), font, font_scale, text_color, thickness, cv2.LINE_AA)

                # Save annotated image
                # Use frame_id for a more descriptive filename
                frame_id_path = Path(metadata.get('frame_id', f'sample_{i:04d}.png'))
                # Replace slashes with underscores for filename compatibility
                safe_filename = str(frame_id_path).replace('/', '_').replace('\\', '_')
                output_filename = output_dir / f"{Path(safe_filename).stem}_pred.png"

                cv2.imwrite(str(output_filename), image)

            else:
                print(f"Warning: Image file not found or not specified for sample {i} ({metadata.get('frame_id')})")

            if (i + 1) % 5 == 0: # Print progress every 5 images
                 print(f"Processed {i+1}/{len(test_loader.dataset)} samples...")

    print("Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions on test images.")
    parser.add_argument('--model_path', type=str,
                        default='weights/pipelineA/dgcnn_20250405_145031/model_best.pt',
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--data_root', type=str, default=BASE_DATA_DIR,
                        help='Root directory of the dataset.')
    parser.add_argument('--output_dir', type=str,
                        default='results/pipelineA/test_set_visualizations',
                        help='Directory to save annotated images.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA if available.')
    parser.add_argument('--no_cuda', action='store_false', dest='use_cuda',
                        help='Do not use CUDA.')

    args = parser.parse_args()
    visualize_predictions(args)
