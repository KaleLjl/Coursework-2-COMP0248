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

**Important:** All training parameters are now controlled exclusively via `src/pipelineA/config.py`. Command-line arguments are no longer used.

**Steps:**

1.  **Configure `src/pipelineA/config.py`**:
    *   Set `MODEL_PARAMS` (e.g., `model_type`, `k`, `emb_dims`, `dropout`).
    *   Set `TRAIN_PARAMS` (e.g., `num_epochs`, `batch_size`, `learning_rate`, `weight_decay`).
    *   Set `AUGMENTATION_PARAMS` (e.g., `enabled`, rotation, jitter, scale ranges).
    *   Set general parameters like `SEED`, `DEVICE`, `NUM_WORKERS`.
    *   Optionally set `EXP_NAME` for organizing outputs.
2.  **Run Training**:
    ```bash
    python src/pipelineA/training/train.py
    ```

Training outputs (checkpoints, logs) will be saved to directories specified in `config.py` (e.g., `WEIGHTS_DIR`, `LOGS_DIR`) under a timestamped subfolder or the `EXP_NAME` if provided.

#### TensorBoard Visualization

To monitor training progress in real-time using TensorBoard:

1. Install TensorBoard if not already installed:
```bash
pip install tensorboard
```

2. Start TensorBoard by pointing it to the logs directory:
```bash
tensorboard --logdir=logs/pipelineA
```

3. Open your web browser and navigate to:
```
http://localhost:6006
```

TensorBoard provides interactive visualizations of:
- Training and validation loss curves
- Accuracy metrics over time
- F1-score progression
- Precision and Recall values
- Learning rate changes

The logs are automatically saved during training in the `logs/pipelineA` directory with timestamps, allowing you to compare different training runs.

Training and evaluation results are saved to:
- `weights/pipelineA/`: Model checkpoints
- `logs/pipelineA/`: Training logs and metrics
- `results/pipelineA/`: Evaluation metrics and visualizations


### 2. Evaluating a Trained Model

**Important:** All evaluation parameters are now controlled exclusively via `src/pipelineA/config.py`. Command-line arguments are no longer used.

**Steps:**

1.  **Configure `src/pipelineA/config.py`**:
    *   Set `EVAL_CHECKPOINT`: Path to the model checkpoint file (e.g., `weights/pipelineA/dgcnn_YYYYMMDD_HHMMSS/model_best.pt`).
    *   Set `EVAL_TEST_SET`: Which test set to use (1 for Harvard-Subset2, 2 for UCL dataset).
    *   Set `EVAL_BATCH_SIZE`: Batch size for evaluation.
    *   Ensure `MODEL_PARAMS` match the configuration used for the checkpoint being evaluated.
    *   Set `DEVICE`.
2.  **Run Evaluation**:
    ```bash
    python src/pipelineA/training/evaluate.py
    ```

Evaluation results (metrics, potentially plots) will be saved to the directory specified in `config.py` (e.g., `RESULTS_DIR`).

### 3. Visualizing Test Predictions on Images

**Important:** All visualization parameters are now controlled exclusively via `src/pipelineA/config.py`. Command-line arguments are no longer used.

**Steps:**

1.  **Configure `src/pipelineA/config.py`**:
    *   Set `EVAL_CHECKPOINT`: Path to the model checkpoint file to use for visualization.
    *   Set `EVAL_TEST_SET`: Which test set to visualize predictions for (1 for Harvard-Subset2, 2 for UCL dataset).
    *   Set `VIS_OUTPUT_DIR`: Directory where annotated images will be saved.
    *   Ensure `MODEL_PARAMS` match the configuration used for the checkpoint.
    *   Set `DEVICE`, `NUM_WORKERS`.
2.  **Run Visualization**:
    ```bash
    python src/pipelineA/visualize_test_predictions.py
    ```

The script will generate images in the specified `VIS_OUTPUT_DIR`, showing the RGB image annotated with "Predicted: [Label]" and "Ground Truth: [Label]". The text color indicates correctness (green for correct, red for incorrect).

### 4. Testing Individual Components

*(Note: The test script might need updates to align with current code.)*

To validate pipeline components:

```bash
python src/pipelineA/test_pipeline.py
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

