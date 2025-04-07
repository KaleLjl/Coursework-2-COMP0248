import os
import sys
# import argparse # Removed argparse import
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to sys.path to allow importing modules from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.pipelineA.data_processing.dataset import TableDataset, collate_fn, create_data_loaders
from src.pipelineA.models.classifier import get_model # Import model factory
from src.pipelineA.models.utils import load_checkpoint # Import checkpoint loader
from src.pipelineA.config import (
    BASE_DATA_DIR, TEST_FRAMES, UCL_DATA_CONFIG, # Removed REAL_SENSE_SEQUENCES, Import dataset specs
    MODEL_PARAMS, POINT_CLOUD_PARAMS,
    # Import visualization/evaluation parameters from config
    EVAL_CHECKPOINT, EVAL_TEST_SET, VIS_OUTPUT_DIR, # Use EVAL_CHECKPOINT and EVAL_TEST_SET
    # General config params
    DEVICE, NUM_WORKERS
)

def visualize_predictions():
    """Loads a model, runs inference on the test set, and saves annotated images. Reads config from config.py."""

    # --- Configuration ---
    # Use DEVICE from config, check CUDA availability
    use_cuda = DEVICE.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(DEVICE if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Use VIS_OUTPUT_DIR from config
    output_dir = Path(VIS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # --- Load Model ---
    # Use MODEL_PARAMS from config.py for consistency
    model_type = MODEL_PARAMS.get('model_type', 'dgcnn')
    print(f"Using model type from config: {model_type}")

    # Instantiate model using the factory function and config parameters
    model = get_model(
        model_type=model_type,
        num_classes=MODEL_PARAMS.get('num_classes', 2), # Default to 2 classes
        k=MODEL_PARAMS.get('k', 20),
        emb_dims=MODEL_PARAMS.get('emb_dims', 1024),
        dropout=MODEL_PARAMS.get('dropout', 0.5),
        feature_dropout=MODEL_PARAMS.get('feature_dropout', 0.0)
    ).to(device)

    # Load checkpoint using the utility function and EVAL_CHECKPOINT from config
    model_path = Path(EVAL_CHECKPOINT) # Use EVAL_CHECKPOINT
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path} (specified in config as EVAL_CHECKPOINT)")
        return

    try:
        # Pass None for optimizer as we are only evaluating
        model, _, _, _ = load_checkpoint(model, None, str(model_path))
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading checkpoint from {model_path}: {e}")
        return

    model.eval() # Set model to evaluation mode

    # --- Load Test Data ---
    # Use EVAL_TEST_SET from config
    test_set_to_use = EVAL_TEST_SET # Use EVAL_TEST_SET
    print(f"Visualizing Test Set: {test_set_to_use} (from config EVAL_TEST_SET)") # Use EVAL_TEST_SET

    # Select the appropriate data specification based on the test set
    if test_set_to_use == 1:
        data_spec = TEST_FRAMES
        dataset_name = "Test Set 1 (Harvard)"
        if not data_spec:
             print("Warning: TEST_FRAMES list is empty in config. Cannot visualize Test Set 1.")
             return
    elif test_set_to_use == 2:
        data_spec = UCL_DATA_CONFIG
        dataset_name = f"Test Set 2 ({UCL_DATA_CONFIG.get('name', 'UCL')})" # Updated name
        if not data_spec:
             print("Warning: UCL_DATA_CONFIG is not defined or empty in config. Cannot visualize Test Set 2.") # Updated message
             return
    else:
        print(f"Error: Invalid test set specified in config (EVAL_TEST_SET): {test_set_to_use}. Choose 1 or 2.") # Use EVAL_TEST_SET, updated range
        return

    print(f"Loading data for: {dataset_name}")

    # Use create_data_loaders with the selected data_spec and config parameters
    _, _, test_loader = create_data_loaders(
        data_root=BASE_DATA_DIR, # Use BASE_DATA_DIR from config
        test_spec=data_spec,
        batch_size=1, # Process one image at a time for visualization
        num_workers=NUM_WORKERS, # Use NUM_WORKERS from config
        point_cloud_params=POINT_CLOUD_PARAMS
    )

    if not test_loader or len(test_loader.dataset) == 0:
        print(f"Error: Failed to create or test data loader is empty for {dataset_name}.")
        return

    print(f"Loaded {len(test_loader.dataset)} test samples for {dataset_name}.")

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
    # Removed argparse setup
    visualize_predictions() # Call function directly
