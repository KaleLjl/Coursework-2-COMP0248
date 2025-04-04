import os
from pathlib import Path

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset paths
BASE_DATA_DIR = PROJECT_ROOT / "data" / "CW2-Dataset" / "data"

# Training sequences (MIT)
TRAIN_SEQUENCES = {
    "mit_32_d507": ["d507_2"],
    "mit_76_459": ["76-459b"],
    "mit_76_studyroom": ["76-1studyroom2"],
    "mit_gym_z_squash": ["gym_z_squash_scan1_oct_26_2012_erika"],
    "mit_lab_hj": ["lab_hj_tea_nov_2_2012_scan1_erika"]
}

# Test sequences (Harvard)
TEST1_SEQUENCES = {
    "harvard_c5": ["hv_c5_1"],
    "harvard_c6": ["hv_c6_1"],
    "harvard_c11": ["hv_c11_2"],
    "harvard_tea_2": ["hv_tea2_2"]  # Raw depth
}

# Test 2 sequences (RealSense)
TEST2_SEQUENCES = {}  # To be collected

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
    "model_type": "dgcnn",  # dgcnn, pointnet, point_transformer
    "k": 10,                # k in kNN graph (Reduced from 20)
    "emb_dims": 512,       # Embedding dimensions (Reduced from 1024) - Keeping reduced for now
    "dropout": 0.0,         # Disabling dropout
    "feature_dropout": 0.0, # Disabling feature dropout
    "reduced_model": False, # Whether to use reduced complexity model
}

# Training parameters
TRAIN_PARAMS = {
    "batch_size": 16,
    "num_epochs": 100,      # Restore full epochs for proper training run
    "learning_rate": 0.001, # Keeping standard LR for now
    "weight_decay": 0.0,    # Disabling weight decay
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

# Paths for saving models and results
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "pipelineA"
RESULTS_DIR = PROJECT_ROOT / "results" / "pipelineA"
LOGS_DIR = PROJECT_ROOT / "logs" / "pipelineA"

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
