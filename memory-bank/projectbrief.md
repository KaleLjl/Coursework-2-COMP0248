# Project Brief: Table Detection from 3D Point Clouds

## Project Overview

This project implements a pipeline for detecting tables in 3D point clouds derived from RGBD images:

1. **Pipeline A**: Convert a depth input to a point cloud, then use a classifier to determine if there's a table in the scene.

_(Note: The scope has been reduced to focus solely on Pipeline A)._

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
- **Depth Format**: `harvard_tea_2` contains raw depth maps (`uint16`, likely mm); other sequences use pre-processed DepthTSDF maps (`float32`, likely meters). The custom 'ucl' dataset also uses raw depth.

## Data Split Strategy

The coursework (`CW2.pdf`) defines the following split:

- **Training Data**: MIT sequences (290 RGBD frames)
- **Test Data 1**: Harvard sequences (98 RGBD frames)
- **Test Data 2**: RealSense sequence (max 50 RGBD frames)

To enable model selection and hyperparameter tuning while preserving an unseen test set, the following splits have been used:

- **Original Split (Used for Exp 1 - Best Balanced Results So Far)**:
  - Training: MIT sequences (~290 frames).
  - Validation: Stratified random subset of Harvard sequences (48 frames).
  - Test Set 1: Remaining stratified random subset of Harvard sequences (50 frames).
- **Domain Adaptation Split (Used for Run `dgcnn_20250407_171414`)**:
  - Training: MIT sequences **plus** `harvard_tea_2` (~305 frames).
  - Validation: Stratified random subset of remaining Harvard sequences (24 frames).
  - Test Set 1: Remaining stratified random subset of Harvard sequences (50 frames).
- **Test Set 2 (Consistent Across Splits)**: Custom 'ucl' dataset (RealSense capture, raw depth) - Used for final evaluation (defined by `UCL_DATA_CONFIG` in `config.py`).

**Current Status (2025-04-07)**: The Domain Adaptation Split led to poor evaluation results (prediction bias). The final split strategy is **under review**, pending investigation into class imbalance in the mixed training set. The Original Split currently represents the configuration with the best achieved evaluation metrics.

## Tasks

1. **Binary Classification**: Determine if there's a table in the image using Pipeline A.

## Implementation Requirements

- Models can be re-used/imported from existing sources, but must be cited
- For full marks, models should differ from tutorial examples (vanilla PointNet, MonoDepth2 pre-trained on KITTI, etc.)
- Monocular Depth can use off-the-shelf pre-trained models if they work reasonably well
- Classification/Segmentation models can use pre-trained models as initialization but must be trained on the coursework dataset

## Evaluation

- Quantitative evaluation for classification performance of Pipeline A.
- Identification of strengths and weaknesses of Pipeline A.

## Deliverables

1. **Report** (Focusing _only_ on Pipeline A, approx. 2 pages total + references, maintaining standard section structure):

   - Introduction / Problem Statement
   - Data processing
   - Method (Pipeline A)
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
   │   └── pipelineA/
   ├── data/
   │   └── ucl/          # Include captured data (renamed from RealSense)
   ├── results/          # Predictions, logs, plots for Pipeline A
   ├── requirements.txt
   └── README.md         # Explain structure and how to run
   ```
   - Note: `data/CW2-Dataset` and `weights/` should be deleted before submission.

## Current Progress

Pipeline A is implemented and has undergone initial tuning. Pipelines B and C will not be implemented as per the revised project scope.
