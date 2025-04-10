# Table Classification Using MiDaS-Estimated Depth and ResNet18 - Pipeline B

## Overview

Pipeline B performs the **binary classification** (table/no-table) task using **depth estimated from monocular RGB images**. Unlike Pipeline A, this approach does not rely on camera-captured depth input. Instead, it uses the MiDaS model to estimate depth from RGB data, and then fed depth estimations to a 1-channel modified ResNet18 classifier.

## Running Enviroment
- Python 3.12.9
- PyTorch 2.6.0
- torchvision 0.21.0
- Open3D
- NumPy
- OpenCV
- SciPy
- Matplotlib
- sklearn
- glob
- pandas
- tqdm
- pickle
- csv

## Pipeline Structure

```
src/pipelineB/
├── MiDaS/                                # Clone the MiDaS repository here if having trouble running locally.
│   └── ...         
├── dataloader.py                         # Dataset class for loading .npy depth + labels 
├── evaluate_uclDataset.py                # Evaluates best model on custom UCL dataset
├── MiDaS_depth_model.py                  # Generates depth .npy/.png using MiDaS from RGB images
├── read_labels.py                        # Converts annotations into labels.csv for SUN3D/UCL
├── read_training_metrics.py              # Utility to read best epoch metrics during training
├── ResNet18_classification_model.py      # Trains depth-based ResNet18 classifier
└── visualization_training_results.py     # Utility to visualize training metrics over epochs
```

## Configuration
- **Data processing parameters**: Set to True in `MiDaS_depth_model.py` or `read_labels.py` to generate depth estimations or label.csv for the corresponding dataset.
  - `PROCESS_TRAIN`: Training dataset (MIT), default: False
  - `PROCESS_TEST1`: Validation dataset (Harvard), default: False
  - `PROCESS_TEST2`: Testing dataset (UCL), default: True
- **Training parameters**: 
  - `IF_PRETRAIN`: Batch size for training (default: True)
  - `NUM_EPOCHS`: Number of training epochs (default: 15)
  - `BATCH_SIZE`: Batch size for training (default: 32)

## Model Architectures

Pipeline B implements two models: 

1. **MiDaS**: Monocular depth estimation model.
   - Estimates depth from monocular RGB data.

2. **ResNet18**: retrained on ImageNet, fine-tuned on depth)
   - Modified to take 1-channel depth input and output binary classification,
   - Pretrained on ImageNet,
   - Fine-tuned on CW2 datasets.

## Usage
### 1. Dataset Structure

Place the CW2 SUN3D datasets and the UCL dataset as shown in this structure (or change path configs accordingly):

```
data/
├── CW2-Dataset/                      # CW2 SUN3D datasets
│   └── data/                         
│       ├── harvard_c5/
│       └── ...
└── ucl_dataset/                      # UCL dataset       
```

### 2. Depth Estimation
Run the MiDaS model to generate depth estimations for Pipeline B.
```bash
    python src/pipelineB/MiDaS_depth_model.py
```

### 3. Read Labels
Convert annotations into labels.csv for SUN3D/UCL datasets.
```bash
    python src/pipelineB/read_labels.py 
```

### 4. Train Classifier
Edit the training configs as needed and then train and validate the classification model on split SUN3D datasets.
```bash
    python src/pipelineB/ResNet18_classification_model.py
```

### 5. Visualize and Store Training Metrics
Visualize and store training metrics from the best model.
```bash
    python src/pipelineB/visualization_training_results.py
```
```bash
    python src/pipelineB/read_training_metrics.py
```

### 6. Evaluate on UCL Dataset
Evaluate the model on the UCL dataset and store the performance metrics.
```bash
    python src/pipelineB/evaluate_uclDataset.py
```















