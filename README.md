# Table Detection from 3D Point Clouds - Pipeline A

## Overview

Pipeline A is a computer vision system for detecting tables in 3D point clouds derived from RGBD images. The pipeline performs two key operations:
1. Converting depth maps to 3D point clouds
2. Using a deep learning classifier to determine if there's a table in the scene (binary classification)

## Pipeline Structure

```
src/pipelineA/
├── data_processing/
│   ├── depth_to_pointcloud.py   # Converts depth maps to point clouds
│   ├── dataset.py               # Custom dataset for data loading
│   └── preprocessing.py         # Point cloud preprocessing functions
├── models/
│   ├── classifier.py            # Neural network model architectures 
│   └── utils.py                 # Model utility functions
├── training/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── config.py                    # Configuration parameters
├── main.py                      # Main entry point
└── test_pipeline.py             # Testing utilities
```

## Configuration

Key configuration parameters in `config.py`:

- **Dataset paths**: Locations of training (MIT) and test (Harvard) sequences
- **Point cloud parameters**: Number of points, normalization settings, depth thresholds
- **Model parameters**: Architecture type (DGCNN/PointNet), network hyperparameters
- **Training parameters**: Batch size, learning rate, regularization settings
- **Augmentation parameters**: Settings for data augmentation during training

## Model Architectures

Pipeline A implements two point cloud processing architectures:

1. **DGCNN (Dynamic Graph CNN)**: Constructs graphs dynamically in feature space, using EdgeConv operations for better capturing of local geometric structures.
2. **PointNet**: A pioneering architecture for point cloud processing that respects permutation invariance.

## Usage

### 1. Training a Model

```bash
python -m src.pipelineA.main train --model_type dgcnn --num_epochs 100 --batch_size 16 --augment
```

Key training parameters:
- `--model_type`: Model architecture (`dgcnn` or `pointnet`)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--augment`: Enable data augmentation
- `--learning_rate`: Initial learning rate
- `--k`: Number of nearest neighbors for DGCNN (default: 20)

### 2. Evaluating a Trained Model

```bash
python -m src.pipelineA.main evaluate --model_type dgcnn --checkpoint /path/to/checkpoint.pt --test_set 1 --visualize
```

Key evaluation parameters:
- `--model_type`: Model architecture (must match the trained model)
- `--checkpoint`: Path to model checkpoint file
- `--test_set`: Test set to use (1: Harvard, 2: RealSense)
- `--visualize`: Enable visualization of predictions

### 3. Testing Individual Components

To validate pipeline components:

```bash
python -m src.pipelineA.test_pipeline
```

This runs tests for:
- Depth to point cloud conversion
- Dataset loading
- Model creation and forward pass

## Data Processing

The pipeline processes data through these steps:

1. **Depth Map Loading**: Reads depth maps from TSDF-processed or raw depth files
2. **Point Cloud Generation**: Projects depth pixels to 3D using camera intrinsics
3. **Preprocessing**: Normalizes and samples point clouds to fixed size (2048 points)
4. **Augmentation**: Applies random rotations, jitter, and scaling during training

## Results and Metrics

The pipeline tracks the following metrics:
- Accuracy
- Precision, Recall, F1-score
- AUC-ROC
- Confusion matrix

Training and evaluation results are saved to:
- `weights/pipelineA/`: Model checkpoints
- `logs/pipelineA/`: Training logs
- `results/pipelineA/`: Evaluation metrics and visualizations
