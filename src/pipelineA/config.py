import os
from pathlib import Path

# Dataset paths
BASE_DATA_DIR = Path("/cs/student/projects1/rai/2024/jialeli/Objection-Coursework2/data/CW2-Dataset/data")

# Training sequences (MIT)
TRAIN_SEQUENCES = {
    "mit_32_d507": ["d507_2"],
    "mit_76_459": ["76-459b"],
    "mit_76_studyroom": ["76-1studyroom2"],
    "mit_gym_z_squash": ["gym_sq1"],  # No tables
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
    "sampling_method": "fps",  # fps, random, or None
    "max_depth": 10.0,   # Maximum depth value in meters
    "min_depth": 0.1     # Minimum depth value in meters
}

# Model parameters
MODEL_PARAMS = {
    "model_type": "dgcnn",  # dgcnn, pointnet, point_transformer
    "k": 20,                # k in kNN graph
    "emb_dims": 1024,       # Embedding dimensions
    "dropout": 0.5,         # Dropout rate
}

# Training parameters
TRAIN_PARAMS = {
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
}

# Data augmentation parameters
AUGMENTATION_PARAMS = {
    "enabled": True,
    "rotation_y_range": [-15, 15],  # Degrees
    "jitter_sigma": 0.01,
    "jitter_clip": 0.05,
    "scale_range": [0.8, 1.2],
    "rotation_z": True  # Rotation around Z axis for orientation invariance
}

# Project root directory
PROJECT_ROOT = Path("/cs/student/projects1/rai/2024/jialeli/Objection-Coursework2")

# Paths for saving models and results
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "pipelineA"
RESULTS_DIR = PROJECT_ROOT / "results" / "pipelineA"
LOGS_DIR = PROJECT_ROOT / "logs" / "pipelineA"

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
