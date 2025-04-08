import os
import torch
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
    from pipelineA.config import (
        UCL_DATA_CONFIG, POINT_CLOUD_PARAMS, MODEL_PARAMS,
        EVAL_CHECKPOINT, DEVICE, BASE_DATA_DIR
    )
    # Assuming dataset and model utilities are structured appropriately
    from pipelineA.data_processing.dataset import TableDataset
    from pipelineA.models.classifier import get_model # Corrected import location
    from pipelineA.models.utils import load_checkpoint # load_checkpoint is in utils
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory or the environment is set up correctly.")
    sys.exit(1)

# --- Configuration ---
# Define the frame IDs you want to inspect (relative paths from BASE_DATA_DIR)
# Example: FRAME_IDS_TO_INSPECT = ["ucl/depth/000608.png", "ucl/depth/000609.png"]
FRAME_IDS_TO_INSPECT = [
    # Frame IDs provided by user
    "ucl/depth/000560.png",
    "ucl/depth/000561.png",
    "ucl/depth/000562.png",
    "ucl/depth/000563.png",
    "ucl/depth/000564.png",
    "ucl/depth/000565.png",
    "ucl/depth/000566.png",
    "ucl/depth/000567.png",
    "ucl/depth/000568.png",
    "ucl/depth/000569.png",
    "ucl/depth/000570.png",
    "ucl/depth/000600.png",
    "ucl/depth/000601.png",
    "ucl/depth/000602.png",
    "ucl/depth/000603.png",
    "ucl/depth/000604.png",
    "ucl/depth/000605.png",
    "ucl/depth/000606.png",
    "ucl/depth/000607.png",
    "ucl/depth/000608.png",
    "ucl/depth/000609.png",
    "ucl/depth/000610.png",
]

# Specify which dataset config to use (e.g., UCL_DATA_CONFIG for Test Set 2)
DATASET_CONFIG = UCL_DATA_CONFIG
DATASET_MODE = 'test' # Or 'val' if inspecting validation frames

# --- Script Logic ---

def inspect_predictions(frame_ids_to_inspect):
    """Loads model, dataset, and inspects predictions for specific frames."""

    if not frame_ids_to_inspect:
        print("Error: FRAME_IDS_TO_INSPECT list is empty. Please add frame IDs to inspect.")
        return

    print(f"Starting Frame Prediction Inspection...")
    print(f"Loading model from checkpoint: {EVAL_CHECKPOINT}")
    print(f"Using device: {DEVICE}")
    print(f"Inspecting {len(frame_ids_to_inspect)} frames...")

    # Load Model
    try:
        # Unpack the MODEL_PARAMS dictionary into keyword arguments
        model = get_model(**MODEL_PARAMS)
        # --- Debugging: Print the checkpoint path ---
        checkpoint_path_to_load = str(EVAL_CHECKPOINT) # Explicitly cast to string
        print(f"DEBUG: Attempting to load checkpoint.")
        print(f"DEBUG: Path type: {type(checkpoint_path_to_load)}")
        print(f"DEBUG: Path value: {checkpoint_path_to_load}")
        if checkpoint_path_to_load is None or not isinstance(checkpoint_path_to_load, str):
             print(f"ERROR: Invalid checkpoint path before calling load_checkpoint! Type: {type(checkpoint_path_to_load)}, Value: {checkpoint_path_to_load}")
             return
        # --- End Debugging ---
        # Corrected argument order: model, optimizer (None), path
        model, _, _, _ = load_checkpoint(model, None, checkpoint_path_to_load)
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Dataset
    # Note: We load the full dataset specified by DATASET_CONFIG and then find the indices
    # for the frames we care about. This ensures consistent preprocessing.
    try:
        # Use BASE_DATA_DIR as data_root when creating the dataset instance
        dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            data_spec=DATASET_CONFIG, # Pass the config dict directly for UCL
            mode=DATASET_MODE,
            augment=False, # No augmentation during inspection
            point_cloud_params=POINT_CLOUD_PARAMS,
            augmentation_params=None
        )
        if not dataset.samples:
             print(f"Error: Dataset loaded 0 samples for config: {DATASET_CONFIG}")
             return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Find indices for the specified frame IDs
    indices_to_inspect = []
    frame_id_map = {sample['frame_id']: i for i, sample in enumerate(dataset.samples)}

    for target_frame_id in frame_ids_to_inspect:
        # Normalize the target frame ID format if necessary (e.g., remove leading slashes)
        normalized_target_id = target_frame_id.strip('/')
        if normalized_target_id in frame_id_map:
            indices_to_inspect.append(frame_id_map[normalized_target_id])
        else:
            # Try matching just the filename stem if it's a UCL path
            if normalized_target_id.startswith("ucl/"):
                 target_stem = Path(normalized_target_id).stem
                 found = False
                 for sample_frame_id, index in frame_id_map.items():
                      if sample_frame_id.startswith("ucl/") and Path(sample_frame_id).stem == target_stem:
                           indices_to_inspect.append(index)
                           found = True
                           break
                 if not found:
                      print(f"Warning: Frame ID '{target_frame_id}' not found in the loaded dataset samples.")
            else:
                 print(f"Warning: Frame ID '{target_frame_id}' not found in the loaded dataset samples.")


    if not indices_to_inspect:
        print("Error: None of the specified frame IDs were found in the dataset.")
        return

    # Inspect predictions for the found indices
    print("\n--- Inspection Results ---")
    with torch.no_grad(): # Disable gradient calculations
        for index in indices_to_inspect:
            try:
                sample = dataset[index]
                points = sample['points'].unsqueeze(0).to(DEVICE) # Add batch dim and move to device
                label_gt = sample['label'].item() # Ground truth label
                metadata = sample['metadata']
                frame_id_display = metadata.get('frame_id', f'Index_{index}')

                # Model inference
                output = model(points) # Output shape is likely (1, 2) - logits for class 0 and 1
                # Apply softmax to get probabilities for each class
                probabilities = torch.softmax(output, dim=1)
                # Select the probability for the positive class (index 1)
                score = probabilities[0, 1].item()
                prediction = 1 if score > 0.5 else 0

                print(f"Frame ID: {frame_id_display}")
                print(f"  Ground Truth: {label_gt}")
                print(f"  Raw Score:    {score:.6f}")
                print(f"  Prediction:   {prediction}")
                print("-" * 20)

            except Exception as e:
                print(f"Error processing index {index} (Frame ID might be {metadata.get('frame_id', 'N/A')}): {e}")

    print("Inspection complete.")


if __name__ == "__main__":
    # Ensure the script is run from the project root for correct path resolution
    os.chdir(PROJECT_ROOT)
    inspect_predictions(FRAME_IDS_TO_INSPECT)
