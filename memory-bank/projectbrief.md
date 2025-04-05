# Project Brief: Table Detection from 3D Point Clouds

## Project Overview
This project implements three different pipelines for detecting tables in 3D point clouds derived from RGBD images:

1. **Pipeline A**: Convert a depth input to a point cloud, then use a classifier to determine if there's a table in the scene.
2. **Pipeline B**: First estimate depth from a 2D RGB image, then use a classifier on the estimated depth map.
3. **Pipeline C**: Convert a depth input to point cloud, then use a segmentation model to classify each point as table or background.

## Dataset
The project uses selected data from Sun 3D dataset, which includes:
- Depth maps
- RGB images
- Table polygon annotations

The labels grouped under a single "table" class include:
- Table top
- Dining table
- Desk
- Coffee table

The project excludes cabinets and kitchen counters from the table class.

### Dataset Notes (from CW2.pdf)
- **Negative Sample Sequences** (No tables): `mit_gym_z_squash`, `harvard_tea_2`
- **Frames with Detected Missing Labels**:
    - `76-1studyroom2 - 0002111-000070763319.jpg`
    - `mit_32_d507 - 0004646-000155745519.jpg`
    - `harvard_c11 - 0000006-000000187873.jpg`
    - `mit_lab_hj - 0001106-000044777376.jpg`
    - `mit_lab_hj - 0001326-000053659116.jpg`
- **Depth Format**: `harvard_tea_2` contains raw depth maps; other sequences use pre-processed DepthTSDF maps.

## Data Split Strategy

The coursework (`CW2.pdf`) defines the following split:
- **Training Data**: MIT sequences (290 RGBD frames)
- **Test Data 1**: Harvard sequences (98 RGBD frames)
- **Test Data 2**: RealSense sequence (max 50 RGBD frames)

To enable model selection and hyperparameter tuning while preserving an unseen test set, the current implementation uses the following split:
- **Training Data**: MIT sequences (`mit_32_d507`, `mit_76_459`, `mit_76_studyroom`, `mit_gym_z_squash`, `mit_lab_hj`) - 290 frames.
- **Validation Data**: A stratified random subset of 48 frames from the Harvard sequences. Used during training for monitoring performance and guiding decisions (e.g., early stopping, model saving).
- **Test Data 1**: The remaining stratified random subset of 50 frames from the Harvard sequences. Used for final evaluation after training is complete.
- **Test Data 2**: RealSense sequence (max 50 RGBD frames) - To be collected and used for final evaluation.

## Tasks
1. **Binary Classification**: Determine if there's a table in the image
2. **Binary Point Cloud Segmentation**: Classify each point as Table or Background

## Implementation Requirements
- Models can be re-used/imported from existing sources, but must be cited
- For full marks, models should differ from tutorial examples (vanilla PointNet, MonoDepth2 pre-trained on KITTI, etc.)
- Monocular Depth can use off-the-shelf pre-trained models if they work reasonably well
- Classification/Segmentation models can use pre-trained models as initialization but must be trained on the coursework dataset

## Evaluation
- Quantitative evaluation for classification/depth estimation
- Qualitative evaluation for segmentation
- Comparison of performance between pipelines
- Identification of strengths and weaknesses

## Deliverables
1. **Report** (maximum 6 pages + references):
   - Introduction / Problem Statement
   - Data processing
   - Methods (Pipelines A, B, C)
   - Results
   - Discussion
   - Conclusions
   - Discussion (~0.75 pages)
   - Conclusions (~0.25 pages)
   - References (no page limit)

2. **Code** (Submitted as `coursework2_groupXX.zip` containing):
   ```
   Code/
   ├── src/
   │   ├── pipelineA/
   │   ├── pipelineB/
   │   └── pipelineC/
   ├── data/
   │   └── RealSense/    # Include captured data
   ├── results/          # Predictions, logs, plots
   ├── requirements.txt
   └── README.md         # Explain structure and how to run
   ```
   - Note: `data/CW2-Dataset` and `weights/` should be deleted before submission.

## Current Progress
Pipeline A is partially implemented, while Pipelines B and C are pending implementation.
