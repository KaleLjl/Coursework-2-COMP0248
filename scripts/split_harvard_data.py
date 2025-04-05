import os
import sys
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Input and output file paths
LABELS_FILE = PROJECT_ROOT / "scripts" / "harvard_frame_labels.pkl"
VAL_FRAMES_OUTPUT = PROJECT_ROOT / "scripts" / "validation_frames.pkl"
TEST_FRAMES_OUTPUT = PROJECT_ROOT / "scripts" / "test_frames.pkl"

# Desired split sizes
N_TOTAL = 98
N_TEST = 50
N_VAL = N_TOTAL - N_TEST # Should be 48

def perform_split():
    """
    Loads frame labels and performs a stratified random split into
    validation and test sets (48 validation, 50 test).
    Saves the resulting frame ID lists.
    """
    # Load the extracted labels
    if not LABELS_FILE.exists():
        print(f"Error: Labels file not found at {LABELS_FILE}")
        print("Please run extract_harvard_labels.py first.")
        sys.exit(1)
        
    try:
        with open(LABELS_FILE, 'rb') as f:
            frame_labels_data = pickle.load(f)
        print(f"Loaded {len(frame_labels_data)} labels from {LABELS_FILE}")
    except Exception as e:
        print(f"Error loading labels file: {e}")
        sys.exit(1)

    if len(frame_labels_data) != N_TOTAL:
         print(f"Warning: Expected {N_TOTAL} labels, but found {len(frame_labels_data)}. Proceeding anyway.")
         # Adjust N_TEST if needed, though ideally it matches N_TOTAL
         current_total = len(frame_labels_data)
         # Keep N_TEST=50 if possible, adjust N_VAL
         n_test_adjusted = min(N_TEST, current_total - 1) # Need at least 1 for validation
         n_val_adjusted = current_total - n_test_adjusted
         print(f"Adjusting split to {n_val_adjusted} validation, {n_test_adjusted} test.")
    else:
        n_val_adjusted = N_VAL
        n_test_adjusted = N_TEST
        print(f"Proceeding with split: {n_val_adjusted} validation, {n_test_adjusted} test.")


    # Prepare lists for scikit-learn
    frame_ids = [item['frame_id'] for item in frame_labels_data]
    labels = np.array([item['label'] for item in frame_labels_data])

    # Perform the stratified split
    # We want 50 test samples, so test_size = 50 / total_samples
    # train_test_split splits into train/test. We'll call the 'train' part 'validation'.
    try:
        val_frame_ids, test_frame_ids, _, _ = train_test_split(
            frame_ids,
            labels, # Need labels for stratification
            test_size=n_test_adjusted, # Specify number of test samples
            stratify=labels,
            random_state=42 # for reproducibility
        )
    except Exception as e:
        print(f"Error during train_test_split: {e}")
        # This can happen if a class has too few members for the split
        print("Attempting split without stratification...")
        try:
             val_frame_ids, test_frame_ids = train_test_split(
                 frame_ids,
                 test_size=n_test_adjusted,
                 random_state=42
             )
             print("Performed split without stratification due to error.")
        except Exception as e2:
             print(f"Error during non-stratified split: {e2}")
             sys.exit(1)


    print(f"Split complete: {len(val_frame_ids)} validation frames, {len(test_frame_ids)} test frames.")
    
    # Verify sizes (adjusting for potential rounding if float test_size was used)
    if len(val_frame_ids) != n_val_adjusted or len(test_frame_ids) != n_test_adjusted:
         print("Warning: Final split sizes differ slightly from target.")
         print(f"  Actual Val: {len(val_frame_ids)}, Actual Test: {len(test_frame_ids)}")


    # Save the resulting lists
    try:
        with open(VAL_FRAMES_OUTPUT, 'wb') as f:
            pickle.dump(val_frame_ids, f)
        print(f"Saved validation frame IDs to: {VAL_FRAMES_OUTPUT}")
        
        with open(TEST_FRAMES_OUTPUT, 'wb') as f:
            pickle.dump(test_frame_ids, f)
        print(f"Saved test frame IDs to: {TEST_FRAMES_OUTPUT}")
        
    except Exception as e:
        print(f"Error saving split frame ID lists: {e}")
        sys.exit(1)

if __name__ == "__main__":
    perform_split()
