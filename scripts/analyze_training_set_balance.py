import os
import sys
import pickle
from pathlib import Path
from collections import Counter

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Import TableDataset and BASE_DATA_DIR
    from src.pipelineA.data_processing.dataset import TableDataset
    from src.pipelineA.config import BASE_DATA_DIR
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the project root directory or the PYTHONPATH is set correctly.")
    sys.exit(1)

def analyze_balance():
    """
    Instantiates the TableDataset for the specified training sequences
    (MIT + harvard_tea_2) and analyzes the class balance.
    """
    # Define the training sequences used for the domain adaptation run
    # (Matches TRAIN_SEQUENCES in config.py during that run)
    DOMAIN_ADAPT_TRAIN_SEQUENCES = {
        "mit_32_d507": ["d507_2"],
        "mit_76_459": ["76-459b"],
        "mit_76_studyroom": ["76-1studyroom2"],
        "mit_gym_z_squash": ["gym_z_squash_scan1_oct_26_2012_erika"], # Negative
        "mit_lab_hj": ["lab_hj_tea_nov_2_2012_scan1_erika"],
        "harvard_tea_2": ["hv_tea2_2"] # Negative, Raw Depth
    }

    print(f"Analyzing training sequences from: {BASE_DATA_DIR}")
    print(f"Sequences to process: {list(DOMAIN_ADAPT_TRAIN_SEQUENCES.keys())}")

    try:
        # Instantiate dataset - mode doesn't matter here, augment=False
        # Pass minimal point cloud params to avoid unnecessary processing
        minimal_pc_params = {"num_points": 10} # We only need labels

        training_dataset = TableDataset(
            data_root=BASE_DATA_DIR,
            data_spec=DOMAIN_ADAPT_TRAIN_SEQUENCES,
            augment=False,
            mode='train', # Mode doesn't affect label loading here
            point_cloud_params=minimal_pc_params,
            augmentation_params=None
        )
    except Exception as e:
        print(f"Error initializing TableDataset: {e}")
        sys.exit(1)

    if not training_dataset.samples:
        print("Error: No samples found for the specified training sequences.")
        sys.exit(1)

    total_samples = len(training_dataset.samples)
    print(f"Found {total_samples} total samples in the specified training sequences.")

    # Count labels
    label_counts = Counter(sample['label'] for sample in training_dataset.samples)
    count_0 = label_counts.get(0, 0)
    count_1 = label_counts.get(1, 0)

    print("\n--- Class Balance Analysis ---")
    print(f"Total Samples: {total_samples}")
    print(f"Class 0 (No Table): {count_0} samples")
    print(f"Class 1 (Table):    {count_1} samples")

    if total_samples > 0:
        percent_0 = (count_0 / total_samples) * 100
        percent_1 = (count_1 / total_samples) * 100
        print(f"\nPercentage Class 0: {percent_0:.2f}%")
        print(f"Percentage Class 1: {percent_1:.2f}%")

    print("----------------------------")

if __name__ == "__main__":
    analyze_balance()
