/cs/student/projects1/rai/2024/jialeli/# Table Detection from 3D Point Clouds - Pipeline A

## Overview

Pipeline A is a computer vision system for detecting tables in 3D point clouds derived from RGBD images. The pipeline performs two key operations:
1. Converting depth maps to 3D point clouds
2. Using a deep learning classifier to determine if there's a table in the scene (binary classification)

## Requirements
/cs/student/projects1/rai/2024/jialeli/
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

**Note:** Many training and model parameters (e.g., dropout, weight decay, embedding dimensions, learning rate schedule parameters) are now primarily controlled via `src/pipelineA/config.py`. Command-line arguments override defaults where applicable.

**General Training Command:**

```bash
python src/pipelineA/training/train.py --model_type dgcnn --num_epochs 100 --batch_size 16 --learning_rate 0.001 --augment --k 20 --seed 42 --device cuda
```

**Current Recommended Command ('Augmentation Only' Strategy):**

This command assumes you have configured `src/pipelineA/config.py` for the 'augmentation only' strategy (e.g., `MODEL_PARAMS['dropout'] = 0.0`, `TRAIN_PARAMS['weight_decay'] = 0.0`, `AUGMENTATION_PARAMS['enabled'] = True`).

```bash
python src/pipelineA/training/train.py --model_type dgcnn --augment
```
*(This uses default values for epochs, batch size, learning rate, k, seed, device from `config.py` or the script's defaults, enabling only augmentation via the flag)*

**Key Command-Line Training Arguments:**
- `--model_type`: Model architecture (`dgcnn` or `pointnet`). Default: `dgcnn`.
- `--num_epochs`: Number of training epochs. Default: From `config.py` (e.g., 100).
- `--batch_size`: Batch size for training. Default: From `config.py` (e.g., 16).
- `--learning_rate`: Initial learning rate. Default: From `config.py` (e.g., 0.001).
- `--augment`: Flag to enable data augmentation (defined in `config.py`).
- `--k`: Number of nearest neighbors for DGCNN. Default: From `config.py` (e.g., 20).
- `--num_workers`: Number of data loading workers. Default: 4.
- `--seed`: Random seed for reproducibility. Default: 42.
- `--device`: Device to use (`cuda` or `cpu`). Default: `cuda`.

*(Note: Arguments like `--dropout`, `--weight_decay`, `--emb_dims`, `--train_val_split` are no longer used as command-line arguments and are controlled via `config.py`)*

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

*(Note: The evaluation script might need updates to align with the current training script structure and configuration loading. The command below is based on the previous structure.)*

```bash
# Example - Check src/pipelineA/training/evaluate.py for current usage
python src/pipelineA/training/evaluate.py --model_type dgcnn --checkpoint weights/pipelineA/dgcnn_YYYYMMDD_HHMMSS/model_best.pt
```

Key evaluation parameters (check `evaluate.py` for current arguments):
- `--model_type`: Model architecture (must match the trained model).
- `--checkpoint`: Path to model checkpoint file.
- Potentially others like `--batch_size`, `--device`.

### 3. Testing Individual Components

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
