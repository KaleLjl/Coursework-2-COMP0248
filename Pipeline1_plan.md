# Pipeline A Implementation Plan

## Overview

Pipeline A involves:
1. Converting depth input to a 3D point cloud
2. Using a classifier to determine if there's a table in the image (binary classification)

## 1. Project Structure

```
src/pipelineA/
├── data_processing/
│   ├── __init__.py
│   ├── depth_to_pointcloud.py        # Functions to convert depth maps to point clouds
│   ├── dataset.py                    # Custom dataset class leveraging read_labels.py
│   └── preprocessing.py              # Any necessary preprocessing steps
├── models/
│   ├── __init__.py
│   ├── classifier.py                 # Point cloud classifier architecture
│   └── utils.py                      # Model utility functions
├── training/
│   ├── __init__.py
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation script
├── config.py                         # Configuration settings
└── main.py                           # Main entry point
```

## 2. Data Processing Pipeline

### 2.1 Data Loading and Organization

- **Dataset Structure**:
  - Adapt the provided `read_labels.py` script to create a robust data loader
  - The dataset is organized as:
    ```
    data/CW2-Dataset/data/[sequence_name]/[sub_sequence]/
      ├── intrinsics.txt            # Camera intrinsic parameters
      ├── depthTSDF/                # Preprocessed depth maps
      ├── image/                    # RGB images
      └── labels/
          └── tabletop_labels.dat   # Table polygon annotations
    ```

- **Data Splits**:
  - Training: MIT sequences
    - mit_32_d507
    - mit_76_459
    - mit_76_studyroom
    - mit_gym_z_squash
    - mit_lab_hj
    - Total: 290 RGBD frames
  
  - Test 1: Harvard sequences
    - harvard_c5
    - harvard_c6
    - harvard_c11
    - harvard_tea_2
    - Total: 98 RGBD frames
  
  - Test 2: RealSense captures (to be collected)
    - Maximum 50 RGBD frames
    - Will be more similar to harvard_tea_2 (raw depth) than other sequences

### 2.2 Label Processing

- **Binary Classification Labels**:
  - If any table polygons exist in the frame → label as "Table" (1)
  - If no table polygons exist → label as "No Table" (0)
  - Handle sequences with no tables (negative samples):
    - mit_gym_z_squash
    - harvard_tea_2

### 2.3 Depth to Point Cloud Conversion

- **Load Depth Maps**:
  - Use depthTSDF for all sequences except harvard_tea_2
  - Use raw depth maps for harvard_tea_2
  - Handle potential missing or corrupted depth maps

- **Camera Intrinsics**:
  - Load from intrinsics.txt for each sequence
  - Example from mit_76_459:
    ```
    fx = fy = 570.3422047415297
    cx = 320
    cy = 240
    ```

- **3D Projection Formula**:
  ```
  X = (u - cx) * depth(u,v) / fx
  Y = (v - cy) * depth(u,v) / fy
  Z = depth(u,v)
  ```

- **Point Cloud Filtering**:
  - Remove points with invalid depth (zero or very large values)
  - Optional: Remove outliers using statistical outlier removal

### 2.4 Data Preprocessing

- **Point Cloud Normalization**:
  - Center each point cloud (subtract mean)
  - Scale to unit sphere (divide by maximum distance from center)

- **Point Sampling**:
  - Sample a fixed number of points (1024, 2048, or 4096)
  - If too few points, use random oversampling
  - If too many points, use farthest point sampling (FPS)

- **Data Augmentation**:
  - Random rotation around the vertical (Y) axis (-15° to +15°)
  - Random jitter (small Gaussian noise added to coordinates)
  - Random scaling (0.8x to 1.2x)
  - Random Z-axis rotation for orientation invariance

## 3. Model Selection and Implementation

### 3.1 Model Options

1. **PointNet++**
   - **Advantages**:
     - Hierarchical feature learning
     - Better captures local geometric structures
     - Uses multi-scale grouping for scale invariance
   - **Architecture**:
     - Set abstraction layers (sampling, grouping, PointNet)
     - Feature propagation layers
     - MLP classifiers

2. **DGCNN (Dynamic Graph CNN)**
   - **Advantages**:
     - Constructs graphs dynamically in feature space
     - Captures local geometric structures better
     - Edge convolution operation is permutation invariant
   - **Architecture**:
     - K-nearest neighbor graph construction
     - EdgeConv operations
     - Global pooling
     - MLP classifier

3. **Point Transformer**
   - **Advantages**:
     - Self-attention mechanism for capturing global context
     - State-of-the-art performance on many point cloud tasks
     - Captures long-range dependencies
   - **Architecture**:
     - Vector self-attention
     - Point-wise MLP
     - Hierarchical structure

### 3.2 Recommended Implementation: DGCNN

- **Rationale**:
  - Better at capturing local geometric features than PointNet
  - Less computationally expensive than Point Transformer
  - State-of-the-art performance on classification tasks

- **Architecture Modifications**:
  - Adjust the number of EdgeConv blocks (3 or 4)
  - Final MLP classifier: 512 → 256 → 64 → 1 (binary)
  - Dropout rate of 0.5 before the final layer

- **Loss Function**:
  - Binary cross-entropy loss
  - Optional: Focal loss if class imbalance is significant

- **Optimization**:
  - Adam optimizer with initial learning rate 0.001
  - Learning rate scheduler (reduce on plateau)
  - Weight decay 1e-4 for regularization

## 4. Training Process

### 4.1 Training Loop

- **DataLoader Setup**:
  - Batch size: 16 or 32 (depending on GPU memory)
  - Number of workers: 4 (for parallel data loading)
  - Shuffle: True for training, False for validation

- **Training Loop**:
  - Epochs: 100 with early stopping
  - Validation frequency: Every epoch
  - Early stopping patience: 15 epochs
  - Save best model based on validation F1-score

- **Logging**:
  - TensorBoard for tracking metrics
  - Log training/validation loss, accuracy, precision, recall, F1-score
  - Save model checkpoints every 5 epochs
  - Log confusion matrices periodically

### 4.2 Hyperparameter Tuning

- **Grid Search Parameters**:
  - Learning rate: [0.0001, 0.0005, 0.001]
  - Batch size: [16, 32]
  - Number of points: [1024, 2048]
  - K neighbors (for DGCNN): [10, 20, 40]

- **Validation Strategy**:
  - 80/20 split of MIT sequences for training/validation
  - Alternative: 5-fold cross-validation if dataset size allows

## 5. Evaluation

### 5.1 Test Set Evaluation

- **Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - AUC-ROC
  - Confusion matrix

- **Analysis**:
  - Per-sequence performance analysis
  - Error analysis: identify challenging cases
  - Compare performance on different test sets

### 5.2 RealSense Data Collection and Evaluation

- **Data Collection**:
  - Capture RGBD data using Intel RealSense camera
  - Environments: office, living room, classroom
  - Various table types, lighting conditions, distances
  - Capture both table and no-table scenarios

- **Data Processing**:
  - Convert raw RealSense data to match the dataset format
  - Extract intrinsic parameters from RealSense SDK
  - Generate depth maps and corresponding RGB images

- **Evaluation**:
  - Apply the trained model to RealSense data
  - Compare performance with Harvard test set
  - Analyze performance differences due to camera characteristics

## 6. Visualization and Analysis

### 6.1 Visualization Tools

- **Point Cloud Visualization**:
  - Use Open3D or matplotlib for 3D visualization
  - Color-code points based on classification results
  - Side-by-side comparisons: RGB, depth, point cloud

- **Results Visualization**:
  - Confusion matrices
  - Precision-recall curves
  - ROC curves
  - Class activation maps if applicable

### 6.2 Error Analysis

- **Failure Case Analysis**:
  - Identify patterns in misclassifications
  - Analyze by viewing angle, distance, occlusion
  - Table type analysis (desk vs. dining table)
  - Compare performance across different sequences

- **Performance Analysis**:
  - Model sensitivity to point cloud density
  - Effect of augmentation strategies
  - Impact of hyperparameter choices

## 7. Implementation Plan

### Phase 1: Data Processing
- **Setup project structure**
  - Create directory structure
  - Initialize Python modules
  - Set up configuration files
  - Setup version control

- **Implement depth-to-pointcloud conversion**
  - Create functions to read depth maps
  - Implement camera intrinsics application
  - Develop 3D point projection algorithms
  - Add filtering for invalid depth values

- **Create dataset and dataloader classes**
  - Adapt read_labels.py for data loading
  - Implement custom Dataset class
  - Build DataLoader with batching
  - Add data split functionality

- **Implement data preprocessing and augmentation**
  - Develop point cloud normalization
  - Implement point sampling strategies
  - Create augmentation functions
  - Build preprocessing pipeline

- **Develop visualization utilities**
  - Create 3D point cloud visualizer
  - Implement RGB/depth overlay visualization
  - Add functions for label visualization
  - Build debugging visualization tools

### Phase 2: Model Implementation
- **Implement or adapt chosen classifier (DGCNN)**
  - Import base DGCNN architecture
  - Modify for binary classification task
  - Implement EdgeConv layers
  - Create final classification head

- **Set up training infrastructure**
  - Implement training loop
  - Add validation functionality
  - Create checkpoint saving/loading
  - Set up logging infrastructure

- **Implement evaluation metrics**
  - Create accuracy calculation
  - Implement precision/recall metrics
  - Add F1-score computation
  - Build confusion matrix generation

- **Initial training runs and debugging**
  - Run small-scale training tests
  - Debug data pipeline issues
  - Verify model convergence
  - Check for memory/performance issues

### Phase 3: Training and Refinement
- **Full model training with hyperparameter tuning**
  - Run grid search for hyperparameters
  - Train with various model configurations
  - Identify optimal parameter set
  - Test different learning rate schedules

- **Validation and iterative improvements**
  - Analyze validation performance
  - Identify model weaknesses
  - Implement architectural improvements
  - Test different regularization strategies

- **Error analysis and model refinement**
  - Analyze misclassified examples
  - Identify patterns in errors
  - Refine model based on error analysis
  - Implement potential architectural changes

- **Implement model ensembling if beneficial**
  - Train multiple model variants
  - Implement voting or averaging ensembles
  - Test ensemble performance
  - Compare with single model performance

### Phase 4: RealSense Data and Evaluation
- **Capture RealSense data in various environments**
  - Set up RealSense camera
  - Develop capture script
  - Collect data in different settings
  - Ensure diverse table types and environments

- **Process RealSense data to match dataset format**
  - Extract RGB and depth maps
  - Apply necessary transformations
  - Get camera intrinsics
  - Organize in compatible format

- **Evaluate model on Harvard and RealSense test sets**
  - Run inference on Harvard test set
  - Evaluate on RealSense data
  - Calculate all metrics
  - Compare performance across test sets

- **Generate final metrics and visualizations**
  - Create confusion matrices
  - Generate ROC and PR curves
  - Visualize classification results
  - Create side-by-side comparisons

### Phase 5: Documentation and Report
- **Comprehensive code documentation**
  - Document all functions and classes
  - Add explanatory comments
  - Create usage examples
  - Ensure consistent documentation style

- **Generate final visualizations for report**
  - Create performance graphs
  - Generate point cloud visualizations
  - Prepare comparison figures
  - Make results tables

- **Write detailed report sections**
  - Introduction and problem statement
  - Data processing methodology
  - Model architecture and training
  - Results and evaluation
  - Discussion and conclusions

- **Prepare submission materials**
  - Organize code repository
  - Compile all results
  - Finalize report
  - Package submission files

## 8. Potential Challenges and Solutions

### Challenge 1: Class Imbalance
- **Problem**: Imbalanced distribution of table vs. no-table examples
- **Solutions**:
  - Weighted loss function
  - SMOTE or other oversampling techniques
  - Data augmentation for minority class

### Challenge 2: Point Cloud Density Variation
- **Problem**: Different sequences may have varying point cloud densities
- **Solutions**:
  - Consistent point sampling strategy
  - Normalize point clouds
  - Multi-scale features in model architecture

### Challenge 3: RealSense Data Quality
- **Problem**: RealSense raw depth may have more noise than preprocessed data
- **Solutions**:
  - Apply depth filtering and noise reduction
  - Fine-tune model on small RealSense dataset
  - Test on both raw and filtered RealSense data

### Challenge 4: Computational Requirements
- **Problem**: Point cloud processing can be computationally intensive
- **Solutions**:
  - Optimize data loading pipeline
  - Reduce point cloud size if necessary
  - Batch processing for inference

## 9. Dependencies and Requirements

- Python 3.8+
- PyTorch 1.10+
- Open3D for point cloud processing
- Numpy, Scipy, Scikit-learn
- TensorBoard for visualization
- CUDA-capable GPU with 8GB+ memory
