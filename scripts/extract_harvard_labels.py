import os
import sys
import pickle
from pathlib import Path

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Import TableDataset and BASE_DATA_DIR, but not TEST1_SEQUENCES
    from src.pipelineA.data_processing.dataset import TableDataset
    from src.pipelineA.config import BASE_DATA_DIR
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory or the PYTHONPATH is set correctly.")
    sys.exit(1)

def extract_labels():
    """
    Instantiates the TableDataset for the Harvard sequences (excluding harvard_tea_2)
    and extracts frame identifiers and their binary labels.
    """
    # Define the Harvard sequences to be used for validation/testing explicitly here
    # Excludes harvard_tea_2 which is now part of the training set
    HARVARD_VAL_TEST_SEQUENCES = {
        "harvard_c5": ["hv_c5_1"],
        "harvard_c6": ["hv_c6_1"],
        "harvard_c11": ["hv_c11_2"]
        # "harvard_tea_2": ["hv_tea2_2"] # Excluded
    }

    print(f"Loading Harvard sequences (for Val/Test split) from: {BASE_DATA_DIR}")
    print(f"Sequences to process: {HARVARD_VAL_TEST_SEQUENCES}")

    try:
        # Instantiate dataset - mode doesn't matter here, augment=False
        # Pass minimal point cloud params to avoid unnecessary processing if possible
        minimal_pc_params = {"num_points": 10} # We don't need the points, just the labels

        # Use the locally defined HARVARD_VAL_TEST_SEQUENCES
        # The TableDataset constructor expects 'data_spec' argument now
        harvard_dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            data_spec=HARVARD_VAL_TEST_SEQUENCES, # Pass the correct dict here
            augment=False,
            mode='test', # Mode doesn't affect label loading
            point_cloud_params=minimal_pc_params, # Use minimal params
            augmentation_params=None
        )
    except Exception as e:
        print(f"Error initializing TableDataset: {e}")
        sys.exit(1)

    if not harvard_dataset.samples:
        print("Error: No samples found for the Harvard sequences. Check paths and config.")
        sys.exit(1)
        
    print(f"Found {len(harvard_dataset.samples)} samples in Harvard sequences.")

    frame_labels = []
    for i, sample in enumerate(harvard_dataset.samples):
        # Use the depth file path relative to BASE_DATA_DIR as a unique identifier
        try:
            relative_depth_path = Path(sample['depth_file']).relative_to(BASE_DATA_DIR)
            frame_id = str(relative_depth_path)
        except ValueError:
             # If depth_file is not under BASE_DATA_DIR, use the full path (less ideal)
             frame_id = sample['depth_file']
             
        label = sample['label']
        frame_labels.append({'frame_id': frame_id, 'label': label})
        # Print progress occasionally
        if (i + 1) % 20 == 0:
             print(f"Processed {i+1}/{len(harvard_dataset.samples)} frames...")

    print(f"Finished processing {len(frame_labels)} frames.")
    
    # Save the labels to a file for later use in splitting
    output_file = PROJECT_ROOT / "scripts" / "harvard_frame_labels.pkl"
    os.makedirs(output_file.parent, exist_ok=True)
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(frame_labels, f)
        print(f"Successfully saved frame labels to: {output_file}")
    except Exception as e:
        print(f"Error saving labels to {output_file}: {e}")

    # Optionally print the labels too
    # print("\nFrame Labels:")
    # for item in frame_labels:
    #     print(f"  {item['frame_id']}: {item['label']}")

if __name__ == "__main__":
    extract_labels()
