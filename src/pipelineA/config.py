import os
import pickle
from pathlib import Path

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset paths
BASE_DATA_DIR = PROJECT_ROOT / "data" / "CW2-Dataset" / "data"

# Training sequences (MIT) - Reverted to original MIT-only sequences
TRAIN_SEQUENCES = {
    "mit_32_d507": ["d507_2"],
    "mit_76_459": ["76-459b"],
    "mit_76_studyroom": ["76-1studyroom2"],
    "mit_gym_z_squash": ["gym_z_squash_scan1_oct_26_2012_erika"], # Negative
    "mit_lab_hj": ["lab_hj_tea_nov_2_2012_scan1_erika"]
    # "harvard_tea_2": ["hv_tea2_2"] # Removed from training
}

# Validation and Test Frame Lists (Loaded from pickle files)
# These files are generated by scripts/split_harvard_data.py
VAL_FRAMES_FILE = PROJECT_ROOT / "scripts" / "validation_frames.pkl"
TEST_FRAMES_FILE = PROJECT_ROOT / "scripts" / "test_frames.pkl" # Reverted back

VALIDATION_FRAMES = []
TEST_FRAMES = []

try:
    with open(VAL_FRAMES_FILE, 'rb') as f:
        VALIDATION_FRAMES = pickle.load(f)
    print(f"Loaded {len(VALIDATION_FRAMES)} validation frame IDs from {VAL_FRAMES_FILE}")
except FileNotFoundError:
    print(f"Warning: Validation frames file not found at {VAL_FRAMES_FILE}. Validation set will be empty.")
except Exception as e:
    print(f"Warning: Error loading validation frames from {VAL_FRAMES_FILE}: {e}. Validation set will be empty.")

try:
    with open(TEST_FRAMES_FILE, 'rb') as f:
        TEST_FRAMES = pickle.load(f)
    print(f"Loaded {len(TEST_FRAMES)} test frame IDs from {TEST_FRAMES_FILE}")
except FileNotFoundError:
    print(f"Warning: Test frames file not found at {TEST_FRAMES_FILE}. Test set will be empty.")
except Exception as e:
    print(f"Warning: Error loading test frames from {TEST_FRAMES_FILE}: {e}. Test set will be empty.")


# Original Harvard sequences (commented out, now split into VAL/TEST FRAMES)
# TEST1_SEQUENCES = {
#     "harvard_c5": ["hv_c5_1"],
#     "harvard_c6": ["hv_c6_1"],
#     "harvard_c11": ["hv_c11_2"],
#     "harvard_tea_2": ["hv_tea2_2"]  # Raw depth
# }

# Point cloud parameters
POINT_CLOUD_PARAMS = {
    "num_points": 2048,  # Number of points to sample
    "normalize": True,   # Whether to normalize point cloud
    "sampling_method": "random",  # Changed from fps to random
    "max_depth": 20.0,   # Increased from 10.0 to handle larger distances in harvard_tea_2
    "min_depth": 0.1     # Minimum depth value in meters
}

# Model parameters
MODEL_PARAMS = {
    "model_type": "dgcnn",  # Reverting model type back to DGCNN
    "k": 10,                # k in kNN graph (Reduced from 20)
    "emb_dims": 512,       # Embedding dimensions (Reduced from 1024) - Keeping reduced for now
    "dropout": 0.5,         # Reverting to best dropout value from Exp 1
    "feature_dropout": 0.0, # Disabling feature dropout
    "reduced_model": False, # Whether to use reduced complexity model
}

# Training parameters
TRAIN_PARAMS = {
    "batch_size": 16,
    "num_epochs": 50,      # Restore full epochs for proper training run
    "learning_rate": 0.001, # Keeping standard LR for now
    "weight_decay": 0.0,    # Reverting weight decay for Experiment 3
    "early_stopping_patience": 100,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "gradient_clip": 0.0,   # Disabling gradient clipping
    "mixup_alpha": 0.0,     # Ensure mixup is disabled
}

# Data augmentation parameters
AUGMENTATION_PARAMS = {
    "enabled": True,        # DIAGNOSTIC STEP: Re-enabling augmentation
    "rotation_y_range": [-30, 30],   # Increased from [-15, 15] for more rotation variation
    "jitter_sigma": 0.015,           # Increased from 0.01 for more noise variation
    "jitter_clip": 0.05, 
    "scale_range": [0.75, 1.25],     # Increased from [0.8, 1.2] for more scale variation
    "rotation_z": True,              # Rotation around Z axis for orientation invariance
    "point_dropout_ratio": 0.1,      # Randomly drop this ratio of points (simulates occlusion)
    "random_subsample": True,        # Apply random subsampling during training
    "subsample_range": [0.7, 0.95],  # Range for random subsampling ratio
}

# General Training/Evaluation Settings (Previously CLI args or parser defaults)
SEED = 42
DEVICE = "cuda" # "cuda" or "cpu"
NUM_WORKERS = 4
EXP_NAME = None # Optional experiment name for logging/checkpointing
AUGMENT = AUGMENTATION_PARAMS["enabled"] # Control augmentation based on AUGMENTATION_PARAMS

# Configuration for the custom UCL dataset
UCL_DATA_CONFIG = {
    'base_path': os.path.join(BASE_DATA_DIR, 'ucl'), # Path to data/CW2-Dataset/data/ucl
    'label_file': os.path.join(BASE_DATA_DIR, 'ucl', 'labels', 'ucl_labels.txt'), # Path to your label file
    'name': 'ucl' # Identifier
}

# Evaluation Parameters (used by evaluate.py and visualize_test_predictions.py)
# Updated EVAL_CHECKPOINT to the best model from the latest Exp 1 re-run (trained on MIT only)
EVAL_CHECKPOINT = str(PROJECT_ROOT / "weights" / "pipelineA" / "dgcnn_20250407_193904" / "model_best.pt")
# Specifies the test set to use for evaluation and visualization.
# 1: Harvard subset (defined by TEST_FRAMES loaded from test_frames.pkl - original 50 frames)
# 2: UCL custom dataset (RealSense capture, defined by UCL_DATA_CONFIG)
EVAL_TEST_SET = 2 # Set to Harvard subset (Test Set 1) for overfitting analysis
EVAL_MODEL_TYPE = MODEL_PARAMS['model_type'] # Use model type from MODEL_PARAMS
EVAL_K = MODEL_PARAMS.get('k', 20)           # Use k from MODEL_PARAMS
EVAL_BATCH_SIZE = TRAIN_PARAMS['batch_size'] # Default to training batch size
EVAL_VISUALIZE = False # Whether evaluate.py should also visualize samples
EVAL_NUM_VISUALIZATIONS = 5 # Number of samples to visualize if EVAL_VISUALIZE is True
EVAL_RESULTS_DIR = None # Optional: Specify a different directory for evaluation results, otherwise uses RESULTS_DIR

# Visualization Script Parameters (used by visualize_test_predictions.py)
# Note: Uses EVAL_CHECKPOINT and EVAL_TEST_SET from above
VIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "pipelineA" / "test_set_visualizations" # Specific dir for visualization script output

# Paths for saving models and results
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "pipelineA"
RESULTS_DIR = PROJECT_ROOT / "results" / "pipelineA"
LOGS_DIR = PROJECT_ROOT / "logs" / "pipelineA"

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
