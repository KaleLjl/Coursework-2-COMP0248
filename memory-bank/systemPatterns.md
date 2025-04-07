# System Patterns: Table Detection from 3D Point Clouds

## Overall Architecture

The project follows a modular architecture with three distinct pipelines, each representing a different approach to table detection:

```mermaid
graph TD
    subgraph "Pipeline A"
        A1[Depth Image] --> A2[Point Cloud Conversion]
        A2 --> A3[Point Cloud Classification]
        A3 --> A4[Table/No Table]
    end
    
    subgraph "Pipeline B"
        B1[RGB Image] --> B2[Depth Estimation]
        B2 --> B3[Depth Classification]
        B3 --> B4[Table/No Table]
    end
    
    subgraph "Pipeline C"
        C1[Depth Image] --> C2[Point Cloud Conversion]
        C2 --> C3[Point Cloud Segmentation]
        C3 --> C4[Pointwise Table/Background]
    end
```

## Common Design Patterns

1. **Pipeline Pattern**: Each solution follows a sequential processing pipeline where the output of one component serves as input to the next.

2. **Factory Pattern**: Model creation is abstracted through factory methods that instantiate the appropriate model architecture based on configuration parameters.

3. **Strategy Pattern**: Different processing strategies (e.g., sampling methods, augmentation techniques) can be selected through configuration.

4. **Repository Pattern**: Data access is abstracted through dataset classes that handle loading and preprocessing of different data types.

5. **Observer Pattern**: Training progress is monitored through callbacks that report metrics to TensorBoard for visualization.

## Pipeline A: Depth to Point Cloud Classification

### Components

1. **Data Processing**:
   - `depth_to_pointcloud.py`: Converts depth maps to 3D point clouds
   - `dataset.py`: Handles data loading and preprocessing. Now supports multiple `data_spec` formats: sequence dict (train), frame lists (val/test1), and custom dataset config dict (e.g., for 'ucl' dataset, loading labels from text file).
   - `preprocessing.py`: Contains point cloud preprocessing functions
   - Data split: MIT=Train, Harvard-Subset1=Validation, Harvard-Subset2=Test1, UCL=Test2.

2. **Models**:
   - `classifier.py`: Implements neural network architectures for point cloud classification
   - `utils.py`: Provides utility functions for model operations

3. **Training**:
   - `train.py`: Handles model training and validation.
   - `evaluate.py`: Evaluates model performance on specified test sets (Test Set 1: Harvard, Test Set 2: UCL). Reads all configuration from `config.py`.
   - Training uses MIT sequences, validation uses Harvard-Subset1.

4. **Configuration**:
   - `config.py`: Centralizes ALL configuration parameters (data paths, model params, training settings, evaluation settings, visualization settings, dataset specs). Used by `train.py`, `evaluate.py`, `visualize_test_predictions.py`, and `dataset.py`.
   - Loads validation/test frame lists from pickle files.
   - Contains general settings like `SEED`, `DEVICE`, `NUM_WORKERS`.
   - Contains specific settings for evaluation (`EVAL_*`) and visualization (`VIS_*`).

5. **Visualization**:
   - `visualize_test_predictions.py`: Loads a trained model, runs inference on the configured test set, and saves annotated RGB images. Reads all configuration from `config.py`.

### Data Flow

```mermaid
graph LR
    subgraph "Training Pipeline"
    DM[MIT Depth Maps] --> PM[Preprocessing]
    PM --> PCM[Point Cloud Generation]
    PCM --> SM[Sampling/Normalization]
    SM --> AM[Augmentation]
    AM --> MM[Model Input]
    MM --> T[Training]
    end
    
    subgraph "Validation Pipeline"
    DH[Harvard Depth Maps] --> PH[Preprocessing]
    PH --> PCH[Point Cloud Generation]
    PCH --> SH[Sampling/Normalization]
    SH --> AH[Augmentation]
    AH --> MH[Model Input]
    MH --> V[Validation During Training]
    end

    subgraph "Testing Pipeline"
    DT[Harvard Test Subset Depth Maps] --> PT[Preprocessing]
    PT --> PCT[Point Cloud Generation]
    PCT --> ST[Sampling/Normalization]
    ST --> MT[Model Input]
    MT --> TE[Testing]
    end

    T --> M[Trained Model]
    V --> EM_V[Validation Metrics]
    M --> TE
    TE --> EM_T[Test Metrics]

```

### Key Interfaces

1. **Data Loading Interface**:
   ```python
   # Dataset interface
   class PointCloudDataset(Dataset):
       def __init__(self, data_path, split, transform=None)
       def __getitem__(self, idx)
       def __init__(self, data_root, data_spec, transform=None, augment=False, mode='train', ...)
       def __getitem__(self, idx)
       def __len__()
   ```
   
   The `data_spec` parameter now determines the loading strategy:
   - `dict` (starting with 'mit_'/'harvard_'): Standard training sequences.
   - `list`: Specific frame IDs for validation or standard test sets.
   - `dict` (with 'name' key, e.g., 'ucl'): Custom dataset configuration, loading labels from a specified text file.

2. **Model Interface**:
   ```python
   # Model interface
   class PointCloudClassifier(nn.Module):
       def __init__(self, args)
       def forward(self, x)
   ```
   
   The model architecture (e.g., DGCNN, PointNet) and its parameters (`k`, `emb_dims`, `dropout`, `feature_dropout`) are defined based on `config.py` and command-line arguments in both training and evaluation scripts.

3. **Training Interface**:
   ```python
   # Training interface
   def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args)
   def evaluate(model, data_loader, criterion, args) # Can take val_loader or test_loader
   ```
   
   The training process uses MIT sequences (train_loader) and Harvard-Subset1 (val_loader).
   Final evaluation uses the test set specified by `EVAL_TEST_SET` in `config.py` (1: Harvard-Subset2, 2: UCL).

## Pipeline B: RGB to Depth to Classification

### Components

1. **Depth Estimation**: Model to convert RGB images to depth maps
2. **Classification**: Model to classify depth maps for table detection
3. **Data Processing**: Utilities for RGB processing and depth estimation
4. **Training/Evaluation**: Scripts for training and evaluation

### Data Flow

```mermaid
graph LR
    RGB[RGB Image] --> P[Preprocessing]
    P --> DE[Depth Estimation]
    DE --> DP[Depth Preprocessing]
    DP --> C[Classification]
    C --> O[Output: Table/No Table]
```

## Pipeline C: Depth to Point Cloud Segmentation

### Components

1. **Point Cloud Generation**: Convert depth maps to point clouds
2. **Segmentation Model**: Classify each point as table or background
3. **Evaluation**: Metrics and visualization for point cloud segmentation

### Data Flow

```mermaid
graph LR
    D[Depth Map] --> P[Preprocessing]
    P --> PC[Point Cloud Generation]
    PC --> S[Sampling/Normalization]
    S --> A[Augmentation]
    A --> M[Model Input]
    M --> SG[Segmentation]
    SG --> O[Output: Per-point Table/Background]
```

## Cross-Cutting Concerns

1. **Configuration Management**: Fully centralized in `config.py`. All scripts (`train.py`, `evaluate.py`, `visualize_test_predictions.py`, `dataset.py`) read their parameters directly from this file. No command-line arguments are used.
2. **Logging**: TensorBoard for training metrics, matplotlib for visualizations
3. **Error Handling**: Robust handling of invalid depths, empty point clouds
4. **Performance Monitoring**: Tracking of inference time, memory usage
5. **Evaluation Framework**: Consistent metrics across pipelines for fair comparison
6. **Dataset Split Management**:
   - Training: MIT sequences (~290 frames).
   - Validation: Stratified random subset of Harvard sequences (48 frames).
   - Test Set 1: Remaining stratified random subset of Harvard sequences (50 frames).
   - Test Set 2: Custom 'ucl' dataset (RealSense capture, defined in `UCL_DATA_CONFIG` in `config.py`).
   - Validation set used during training for monitoring and model selection.
   - Test Sets 1 or 2 used for final evaluation via `evaluate.py` (controlled by `EVAL_TEST_SET` in `config.py`).
