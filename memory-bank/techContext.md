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
- **Test Set 2**: Custom 'ucl' dataset (RealSense capture, defined by `UCL_DATA_CONFIG` in `config.py`)

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
└── ucl/                  # Custom dataset (Test Set 2)
    ├── depth/            # Raw depth (uint16)
    ├── image/
    ├── intrinsics.txt
    └── labels/
        └── ucl_labels.txt # Custom text labels
```
Validation and Test Set 1 frames are drawn from the Harvard sequences based on pre-generated frame lists (`validation_frames.pkl`, `test_frames.pkl`). Test Set 2 ('ucl') uses its own structure and label file.

### Data Characteristics and Notes
- **Depth Format**: `harvard_tea_2` and the custom 'ucl' dataset use raw depth (`uint16`, likely mm), others use processed DepthTSDF (`float32`, likely meters). Handled in `dataset.py`.
- **Negative Samples**: `mit_gym_z_squash` and `harvard_tea_2` contain no tables. Labels for 'ucl' are defined in `ucl_labels.txt`.
- **Missing Labels**: Specific frames in `76-1studyroom2`, `mit_32_d507`, `harvard_c11`, `mit_lab_hj` are noted in `CW2.pdf` as potentially missing table labels. This is handled by the current label loading logic (frames without labels are treated as negative).

### Data Loading Pipeline

The `TableDataset` class in `dataset.py` handles loading:
- It accepts a `data_spec` argument which determines the loading strategy:
    - **Training (`dict` starting with 'mit_'/'harvard_'):** Loads standard MIT/Harvard sequences based on `TRAIN_SEQUENCES` from `config.py`. Labels are derived from `tabletop_labels.dat`.
    - **Validation/Test Set 1 (`list`):** Loads specific frames based on `VALIDATION_FRAMES` or `TEST_FRAMES` lists from `config.py`. Labels are derived from the original `tabletop_labels.dat` of the corresponding frames.
    - **Test Set 2 ('ucl') (`dict` with 'name'='ucl'):** Loads data based on `UCL_DATA_CONFIG` from `config.py`. It scans the specified `base_path` for depth/image files and loads binary labels from the specified text `label_file`. Assumes raw depth format.
- It scans the relevant sequences or directories based on `data_spec`.
- It loads depth, (optionally) RGB, and intrinsics.
- It determines the binary label (0/1) based on the source (`tabletop_labels.dat` or custom text file).
- If `data_spec` is a list, it filters the loaded samples to include only those matching the specified frame identifiers.
- It converts depth to point clouds (handling raw vs. TSDF) and applies preprocessing/augmentation.

```mermaid
graph TD
    subgraph Data Loading Pipeline
        direction LR
        DSpec[Data Specification (Dict:Train/UCL(Test2), List:Val/Test1)] --> DC[TableDataset Class]

        subgraph Loading Logic based on DSpec Type
            DSpec -- Dict (Train) --> ScanSeq[Scan MIT/Harvard Sequences]
            ScanSeq --> LoadStd[Load Frames (Depth, RGB, Intrinsics, Labels.dat)]
            LoadStd --> SamplesTrain[All Scanned Samples]

            DSpec -- List (Val/Test1) --> ScanSeqList[Scan Relevant Harvard Sequences]
            ScanSeqList --> LoadStdList[Load Frames (Depth, RGB, Intrinsics, Labels.dat)]
            LoadStdList --> FilterList[Filter by Frame List]
            FilterList --> SamplesValTest1[Filtered Samples (Val/Test1)]

            DSpec -- Dict (UCL) --> ScanUCL[Scan UCL Directory]
            ScanUCL --> LoadUCL[Load Frames (Depth, RGB, Intrinsics, ucl_labels.txt)]
            LoadUCL --> SamplesUCL[All UCL Samples (Test2)]
        end

        SamplesTrain --> Process[Process Sample (PC Gen, Preproc, Aug)]
        SamplesValTest1 --> Process
        SamplesUCL --> Process
        Process --> Output[Loader Output (Points, Label, Metadata)]
    end
```

## Configuration System

The system uses a **fully centralized configuration** approach:

1. **Single Source of Truth**: All configuration parameters are defined in `src/pipelineA/config.py`. This includes:
    - Data paths (`BASE_DATA_DIR`, dataset specs like `TRAIN_SEQUENCES`, `VALIDATION_FRAMES`, `TEST_FRAMES`, `UCL_DATA_CONFIG`).
    - Point cloud processing parameters (`POINT_CLOUD_PARAMS`).
    - Model architecture parameters (`MODEL_PARAMS`).
    - Training hyperparameters (`TRAIN_PARAMS`).
    - Data augmentation settings (`AUGMENTATION_PARAMS`).
    - General settings (`SEED`, `DEVICE`, `NUM_WORKERS`, `EXP_NAME`, `AUGMENT`).
    - Evaluation settings (`EVAL_*` variables like `EVAL_CHECKPOINT`, `EVAL_TEST_SET`, `EVAL_BATCH_SIZE`, etc.).
    - Visualization settings (`VIS_*` variables like `VIS_OUTPUT_DIR`).
    - Output paths (`WEIGHTS_DIR`, `RESULTS_DIR`, `LOGS_DIR`).
2. **No Command-Line Arguments**: Scripts (`train.py`, `evaluate.py`, `visualize_test_predictions.py`) no longer accept or parse command-line arguments. They import necessary parameters directly from `config.py`.
3. **Run-Specific Configuration**: While not explicitly stored *in* the checkpoint file itself, the configuration used for a specific run can be inferred from the timestamped checkpoint/log directory names and the state of `config.py` at the time of the run (requires version control).

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
