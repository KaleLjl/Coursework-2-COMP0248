# Pipeline C: Depth to Point Cloud Segmentation

## Overview

Pipeline C converts depth data to point clouds, then implement binary point cloud segmentation (classify each point).

## Requirements
- Python 3.9.21
- torch 2.6.0+cu126
- torchvision 0.21.0+cu126
- Open3D
- NumPy
- OpenCV
- sklearn
- Matplotlib

## Installation

1. Enter this repository:
```bash
cd Code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Structure

```
src/pipelineC/
        ├── __init__.py
        ├── dataloader.py
        ├── evaluate.py
        ├── model.py
        ├── train.py
        ├── utils.py
        └── visualize.py
```

## Usage

### 1. Training on the MIT sequences
```bash
python src/pipelineC/train.py
```

### 2. Testing and evaluating on the Harvard sequences
```bash
python src/pipelineA/training/evaluate.py
```

### 3. Testing and evaluating on the UCL sequences
```bash
python src/pipelineA/training/evaluate.py --test_set 2
```