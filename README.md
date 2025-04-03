# Table Detection from 3D Point Clouds - Pipeline A

## Overview

Pipeline A is a computer vision system for detecting tables in 3D point clouds derived from RGBD images. The pipeline performs two key operations:
1. Converting depth maps to 3D point clouds
2. Using a deep learning classifier to determine if there's a table in the scene (binary classification)

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Open3D
- NumPy
- OpenCV
- SciPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Objection-Coursework2.git
cd Objection-Coursework2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

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
- **Point cloud parameters**: 
  - `num_points`: Number of points to sample (default: 2048)
  - `normalize`: Whether to normalize point cloud (default: True)
  - `sampling_method`: Method for point sampling ('fps', 'random', or None)
  - `max_depth`: Maximum depth value in meters (default: 10.0)
  - `min_depth`: Minimum depth value in meters (default: 0.1)
- **Model parameters**: 
  - `model_type`: Architecture type ('dgcnn' or 'pointnet')
  - `k`: Number of nearest neighbors for DGCNN (default: 20)
  - `emb_dims`: Embedding dimensions (default: 1024)
  - `dropout`: Dropout rate (default: 0.5)
- **Training parameters**: 
  - `batch_size`: Batch size for training (default: 16)
  - `num_epochs`: Number of training epochs (default: 100)
  - `learning_rate`: Initial learning rate (default: 0.001)
  - `weight_decay`: Weight decay for regularization (default: 1e-4)
- **Augmentation parameters**: 
  - `rotation_y_range`: Range of rotation angles around Y axis (default: [-15, 15])
  - `jitter_sigma`: Standard deviation for point jittering (default: 0.01)
  - `jitter_clip`: Maximum absolute jitter (default: 0.05)
  - `scale_range`: Range of scaling factors (default: [0.8, 1.2])

## Model Architectures

Pipeline A implements two point cloud processing architectures:

1. **DGCNN (Dynamic Graph CNN)**: 
   - Constructs graphs dynamically in feature space
   - Uses EdgeConv operations for better capturing of local geometric structures
   - More robust to point cloud variations and noise

2. **PointNet**: 
   - A pioneering architecture for point cloud processing
   - Respects permutation invariance
   - Simpler architecture with fewer parameters

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
- `--train_val_split`: Train/validation split ratio (default: 0.8)

### 2. Evaluating a Trained Model

```bash
python -m src.pipelineA.main evaluate --model_type dgcnn --checkpoint /path/to/checkpoint.pt --test_set 1 --visualize
```

Key evaluation parameters:
- `--model_type`: Model architecture (must match the trained model)
- `--checkpoint`: Path to model checkpoint file
- `--test_set`: Test set to use (1: Harvard, 2: RealSense)
- `--visualize`: Enable visualization of predictions
- `--num_visualizations`: Number of samples to visualize (default: 5)

### 3. Testing Individual Components

To validate pipeline components:

```bash
python -m src.pipelineA.test_pipeline
```

This runs tests for:
- Depth to point cloud conversion
- Dataset loading
- Model creation and forward pass
- Data augmentation
- Point cloud preprocessing

## Data Processing

The pipeline processes data through these steps:

1. **Depth Map Loading**: 
   - Reads depth maps from TSDF-processed or raw depth files
   - Supports multiple file formats (.npy, .png)
   - Handles missing or corrupted depth values

2. **Point Cloud Generation**: 
   - Projects depth pixels to 3D using camera intrinsics
   - Filters out invalid depth values
   - Creates colored point clouds when RGB data is available

3. **Preprocessing**: 
   - Normalizes point clouds to unit sphere
   - Samples to fixed size (2048 points)
   - Supports multiple sampling methods (FPS, random)

4. **Augmentation**: 
   - Random rotations around Y and Z axes
   - Point jittering with Gaussian noise
   - Random scaling
   - Only applied during training

## Results and Metrics

The pipeline tracks the following metrics:
- Accuracy: Overall classification accuracy
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-score: Harmonic mean of precision and recall
- AUC-ROC: Area under the receiver operating characteristic curve
- Confusion matrix: Detailed breakdown of predictions

Training and evaluation results are saved to:
- `weights/pipelineA/`: Model checkpoints
- `logs/pipelineA/`: Training logs and metrics
- `results/pipelineA/`: Evaluation metrics and visualizations

## Troubleshooting

Common issues and solutions:

1. **Empty Point Clouds**:
   - Check depth map file paths in config.py
   - Verify depth map file format and values
   - Adjust min_depth and max_depth thresholds

2. **Training Issues**:
   - Reduce batch size if running out of memory
   - Adjust learning rate if training is unstable
   - Enable data augmentation for better generalization

3. **Evaluation Problems**:
   - Ensure model type matches checkpoint
   - Verify test set paths and labels
   - Check point cloud preprocessing parameters
