# Technical Context: Table Detection from 3D Point Clouds

## Development Environment

**Activation:** Always activate the Python virtual environment before running any scripts to ensure correct dependencies are used. On Linux/macOS, use:
```bash
source .venv/bin/activate 
```

The project is developed in a Python environment with the following requirements:

```
# From requirements.txt
Python 3.8+
PyTorch 1.8+
Open3D
NumPy
OpenCV
SciPy
Matplotlib
```

## Core Technologies

### Point Cloud Processing

- **Open3D**: Library for working with 3D data, particularly point clouds
- **NumPy**: Foundation for numerical operations on arrays and matrices
- **SciPy**: Scientific computing library used for spatial operations and algorithms

### Deep Learning

- **PyTorch**: Deep learning framework for model development and training
- **CUDA**: For GPU acceleration of model training and inference (if available)
- **TensorBoard**: For visualization of training metrics and progress

### Computer Vision

- **OpenCV**: For image processing operations, especially on RGB and depth data
- **Matplotlib**: For visualization of images, point clouds, and results

## Key Technical Components

### Point Cloud Generation

The depth-to-point-cloud conversion utilizes camera intrinsic parameters to project depth pixels into 3D space. It filters depth values based on `min_depth` and `max_depth` defined in `config.py`. Note: `max_depth` was increased from 10.0m to 20.0m to accommodate larger distances observed in the `harvard_tea_2` sequence.

```python
# Pseudo-code for depth to point cloud conversion
def depth_to_pointcloud(depth_map, intrinsics):
    # Create meshgrid of pixel coordinates
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to homogeneous coordinates
    x = (x - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (y - intrinsics[1, 2]) / intrinsics[1, 1]
    
    # Multiply by depth to get 3D coordinates
    z = depth_map
    x = x * z
    y = y * z
    
    # Stack to create point cloud
    points = np.stack((x, y, z), axis=-1)
    
    # Reshape and filter invalid points
    points = points.reshape(-1, 3)
    valid_mask = z.reshape(-1) > 0
    
    return points[valid_mask]
```

### Point Cloud Preprocessing

Point clouds undergo several preprocessing steps:

1. **Filtering**: Removal of invalid points (NaN, infinity, zero depth)
2. **Downsampling**: Uniform or FPS (Farthest Point Sampling) to reduce point count
3. **Normalization**: Scaling to unit sphere and centering
4. **Augmentation**: Random rotation, scaling, and jittering (training only)

### Neural Network Architectures

The project supports two main architectures for point cloud classification:

1. **DGCNN (Dynamic Graph CNN)**:
   - Constructs dynamic graphs in feature space
   - Uses EdgeConv operations for better geometric understanding
   - More robust to point cloud variations

2. **PointNet**:
   - Pioneering architecture for point cloud processing
   - Respects permutation invariance through max pooling
   - Simpler architecture with fewer parameters

## Data Management

### Dataset Split and Organization

The data is split as follows:
- **Training**: MIT sequences (~290 frames)
- **Validation**: Stratified random subset of Harvard sequences (48 frames)
- **Test Set 1**: Remaining stratified random subset of Harvard sequences (50 frames)
- **Test Set 2**: RealSense sequence (max 50 frames, to be collected)

The data is organized by location:
```
data/
├── MIT/
│   ├── mit_32_d507/
│   ├── mit_76_459/
│   ├── mit_76_studyroom/
│   ├── mit_gym_z_squash/ # Negative samples
│   └── mit_lab_hj/
├── Harvard/
│   ├── harvard_c5/
│   ├── harvard_c6/
│   ├── harvard_c11/
│   └── harvard_tea_2/    # Negative samples, raw depth
└── RealSense/
    └── [custom captured data]
```
Validation and Test Set 1 frames are drawn from the Harvard sequences based on pre-generated frame lists (`validation_frames.pkl`, `test_frames.pkl`).

### Data Characteristics and Notes
- **Depth Format**: `harvard_tea_2` uses raw depth (likely mm), others use processed DepthTSDF (likely meters). Handled in `dataset.py`.
- **Negative Samples**: `mit_gym_z_squash` and `harvard_tea_2` contain no tables.
- **Missing Labels**: Specific frames in `76-1studyroom2`, `mit_32_d507`, `harvard_c11`, `mit_lab_hj` are noted in `CW2.pdf` as potentially missing table labels. This is handled by the current label loading logic (frames without labels are treated as negative).

### Data Loading Pipeline

The `TableDataset` class in `dataset.py` handles loading:
- It accepts a `data_spec` argument:
    - For training: A dictionary mapping sequence names to sub-sequences (`TRAIN_SEQUENCES` from `config.py`).
    - For validation/testing: A list of specific frame identifiers (`VALIDATION_FRAMES` or `TEST_FRAMES` loaded from pickle files in `config.py`).
- It scans the relevant sequences based on `data_spec`.
- It loads depth, (optionally) RGB, intrinsics, and label data (`tabletop_labels.dat`).
- It determines the binary label (0/1) based on the presence of table polygons.
- If `data_spec` is a list, it filters the loaded samples to include only those matching the specified frame identifiers.
- It converts depth to point clouds and applies preprocessing/augmentation.

```mermaid
graph TD
    subgraph Data Loading
        direction LR
        DSpec[Data Specification (Dict or List)] --> DC[Dataset Class (TableDataset)]
        DC --> Scan[Scan Relevant Sequences]
        Scan --> Load[Load Frames (Depth, RGB, Labels, Intrinsics)]
        Load --> Filter{Filter by Frame List?}
        Filter -- Yes --> SamplesValTest[Filtered Samples (Val/Test)]
        Filter -- No --> SamplesTrain[All Scanned Samples (Train)]
        SamplesTrain --> Process[Process Sample (PC Gen, Preproc, Aug)]
        SamplesValTest --> Process
        Process --> Output[Loader Output (Points, Label, Metadata)]
    end
```

## Configuration System

The system uses a centralized configuration approach:

1. **Base Configuration**: Default parameters defined in `config.py`
2. **Command-Line Override**: Arguments passed via CLI take precedence
3. **Run-Specific Configuration**: Stored with model checkpoints for reproducibility

## Deployment & Evaluation

### Model Checkpointing

Models are saved during training with the following information:
- Model weights
- Training configuration with dataset split information
- Optimization state
- Best validation metrics
- Enhanced regularization parameters

### Visualization

Results are visualized through:
- 3D interactive point cloud visualizations
- Confusion matrices for classification results
- PR and ROC curves for model performance
- TensorBoard dashboards for training metrics

### Metrics Tracking

The system tracks key metrics:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrix
- Train/validation divergence metrics
- Generalization gap metrics

### Enhanced Regularization

To address overfitting and ensure generalization to the Harvard validation set, the system implements:

1. **Aggressive Dropout**: Increased from 0.5 to 0.7
2. **Feature-Level Dropout**: Additional 0.2 dropout in feature maps
3. **Weight Decay**: Increased from 1e-4 to 5e-4
4. **Gradient Clipping**: Prevents extreme weight updates
5. **Enhanced Data Augmentation**: More aggressive rotations, jitter, and point dropout
6. **Point Dropout**: Simulates occlusion in point clouds
