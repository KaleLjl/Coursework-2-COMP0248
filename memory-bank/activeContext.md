# Active Context: Table Detection from 3D Point Clouds

## Current Focus

_(Scope revised to focus solely on Pipeline A)_

The investigation into potential class imbalance in the mixed training set (MIT + `harvard_tea_2`) used for the domain adaptation run (`dgcnn_20250407_171414`) is complete. The analysis revealed the dataset was **well-balanced** (47.5% Class 0, 52.5% Class 1).

Therefore, class imbalance was not the cause of the poor evaluation results (prediction bias, low AUC) of that model. The domain adaptation attempt by simply mixing data failed.

**Final Decision**: Use model `dgcnn_20250407_174719` (DGCNN, D=0.5, trained on MIT only) and the original data split (MIT=Train, Harvard-Subset1=Val(48), Harvard-Subset2=Test1(50)).

The current focus is **writing the final report**, covering **only Pipeline A** but maintaining the standard section structure, aiming for approximately **2 pages** total length plus references.

## Recent Changes

1.  **Identify Domain Shift**: Analyzed overfitting results and identified the difference between DepthTSDF (training) and raw depth (`harvard_tea_2`, `ucl` test) as a likely cause for poor generalization.
2.  **Modify Training Set**: Moved `harvard_tea_2` (raw depth) sequence to the training set to expose the model to both data domains. Updated `src/pipelineA/config.py`.
3.  **Update Data Split Scripts**:
    - Modified `scripts/extract_harvard_labels.py` to exclude `harvard_tea_2` from the pool used for validation/test splitting.
    - Confirmed `scripts/split_harvard_data.py` handles variable input sizes correctly.
4.  **Regenerate Data Splits**:
    - Executed `scripts/extract_harvard_labels.py` to create labels for the remaining 74 Harvard frames.
    - Executed `scripts/split_harvard_data.py` to create new validation (24 frames) and test (50 frames) sets from the reduced Harvard pool, saving `validation_frames.pkl` and `test_frames.pkl`.
5.  **Execute Training (Domain Adaptation Attempt)**: Ran `src/pipelineA/training/train.py` using the best configuration (DGCNN, D=0.5) on the new mixed training set (MIT + `harvard_tea_2`). Run ID: `dgcnn_20250407_171414`. Best validation F1 (0.9787) achieved at Epoch 1.
6.  **Evaluate Domain Adaptation Model**: Evaluated model `dgcnn_20250407_171414` on the updated Test Set 1 (50 Harvard frames excl. `harvard_tea_2`). Results: Acc: 0.9400, Precision: 0.9400, Recall: 1.0000, F1: 0.9691, AUC: 0.2553.
7.  **Analyze & Decide (Tentative)**: Tentatively concluded that the domain adaptation attempt failed due to prediction bias. Decision to revert was on hold.
8.  **Analyze Training Set Class Balance**: Created and executed `scripts/analyze_training_set_balance.py` for the mixed training set (MIT + `harvard_tea_2`). Found the set to be well-balanced (47.5% Class 0, 52.5% Class 1).
9.  **Final Decision**: Confirmed domain adaptation attempt failed for reasons other than class imbalance. **Final decision made to revert** to the previous best model (`dgcnn_20250407_142213`) and the original data split.

_(Previous changes retained below)_

1.  **Overfitting Analysis Execution**:
    _ Confirmed configuration (`EVAL_CHECKPOINT`, set `EVAL_TEST_SET=1`).
    _ Executed `evaluate.py` and `visualize_test_predictions.py` on Test Set 1 (Harvard-Subset2).
    _ Temporarily modified `config.py` to load validation frames.
    _ Executed `evaluate.py` and `visualize_test_predictions.py` on Validation Set (Harvard-Subset1).
    _ Reverted `config.py`.
    _ Summarized comparison results (see Learnings/Insights).
    be2. **Fix ImportError Post-Cleanup**: Removed the import of `REAL_SENSE_SEQUENCES` from `evaluate.py` and `visualize_test_predictions.py` after it was deleted from `config.py` during the test set consolidation, resolving the `ImportError`. Updated Memory Bank.
2.  **Consolidate Test Sets (Cleanup)**: Modified `config.py`, `evaluate.py`, and `visualize_test_predictions.py` to remove the unused `REAL_SENSE_SEQUENCES` definition and designate the UCL dataset (using `UCL_DATA_CONFIG`) as Test Set 2. Updated relevant comments and logic. Updated Memory Bank files accordingly.
3.  **Initiate Cleanup Phase**: Shifted focus from new pipeline development to project cleanup.
4.  **Remove CLI Args (`train.py`)**: Modified `src/pipelineA/training/train.py` to remove `argparse` and source all parameters from `config.py`. (Used `write_to_file` fallback).
5.  **Remove CLI Args (`evaluate.py`)**: Modified `src/pipelineA/training/evaluate.py` to remove `argparse` and source all parameters from `config.py`.
6.  **Remove CLI Args (`visualize_test_predictions.py`)**: Modified `src/pipelineA/visualize_test_predictions.py` to remove `argparse` and source all parameters from `config.py`.
7.  **Update `config.py`**: Added necessary top-level configuration variables (e.g., `SEED`, `DEVICE`, `NUM_WORKERS`, `AUGMENT`, `EXP_NAME`) and consolidated evaluation/visualization parameters (e.g., `EVAL_CHECKPOINT`, `EVAL_TEST_SET`, `VIS_OUTPUT_DIR`) previously handled by CLI or defaults. Renamed default evaluation parameters for clarity.

_(Previous changes retained below)_

1.  **Update Memory Bank (Post-Exp 5)**: Documented Experiment 5 results and the conclusion.
2.  **Configuration Update (Exp 5)**: Modified `src/pipelineA/config.py` to set `MODEL_PARAMS['model_type'] = 'pointnet'` (keeping D=0.5, WD=0, FD=0).
3.  **Execute Training (Exp 5 - PointNet)**: Ran `src/pipelineA/training/train.py --model pointnet`. Best validation F1: 0.8293, Acc: 0.7083 at Epoch 2. Run ID: `pointnet_20250405_155003`.
4.  **Execute Evaluation (Exp 5 - PointNet)**: Ran `src/pipelineA/training/evaluate.py --model_type pointnet` on the best checkpoint. Results: Acc: 0.7200, Precision: 0.7200, Recall: 1.0000, F1: 0.8372, AUC: 0.4226. Poor performance.
5.  **Create Visualization Script**: Created `src/pipelineA/visualize_test_predictions.py` to load the best model, run inference on Test Set 1, and save annotated RGB images showing predictions vs ground truth to `results/pipelineA/test_set_visualizations/`.
6.  **Enable Custom Dataset Evaluation ('ucl')**:
    - Added `UCL_DATA_CONFIG` to `src/pipelineA/config.py` specifying the path to the 'ucl' dataset and its text label file.
    - Modified `src/pipelineA/data_processing/dataset.py` (`TableDataset._load_and_filter_samples`, added `_load_ucl_dataset`) to handle loading data based on `UCL_DATA_CONFIG` and reading labels from the specified text file.
    - Modified `src/pipelineA/training/evaluate.py` to accept `--test_set 2` (previously 3) and use the `UCL_DATA_CONFIG` to load and evaluate the 'ucl' dataset.
7.  **Enable Default Evaluation Run**:
    - Added default evaluation parameters (`DEFAULT_EVAL_CHECKPOINT`, `DEFAULT_EVAL_TEST_SET`, etc.) to `src/pipelineA/config.py`.
    - Modified `src/pipelineA/training/evaluate.py` to check `sys.argv`. If no command-line arguments are provided, it uses the default parameters from `config.py` instead of requiring CLI input.
8.  **Align Visualization Script (Initial)**: Modified `src/pipelineA/visualize_test_predictions.py` to align with `evaluate.py`. It now uses the `get_model` factory and `load_checkpoint` utility, sourcing model parameters and the default checkpoint path from `config.py` for consistency. Fixed a `TypeError` in the `load_checkpoint` call.
9.  **Align Visualization Script (Dataset Selection)**: Further modified `src/pipelineA/visualize_test_predictions.py` to include an optional `--test_set` argument (no longer used due to config centralization). If omitted, the script now defaults to visualizing the dataset specified by `EVAL_TEST_SET` in `config.py`, matching the default evaluation behavior. `EVAL_TEST_SET` selects Test Set 1 (Harvard) or 2 (UCL).
10. **Fix Visualization Script Config Usage**: Updated `src/pipelineA/visualize_test_predictions.py` to use the correct centralized configuration variables (`EVAL_CHECKPOINT`, `EVAL_TEST_SET`) instead of the removed `VIS_MODEL_PATH` and `VIS_TEST_SET`.
11. **Verify Pipeline A Workflow**: Successfully executed `train.py` (briefly), `evaluate.py`, and the fixed `visualize_test_predictions.py` using the centralized configuration, confirming the workflow functions correctly after the _initial_ cleanup.
12. **Fix Visualization Image Matching (Attempt 1)**: Updated `src/pipelineA/data_processing/dataset.py` to match depth and image files based on frame number (ignoring timestamps). This resolved warnings for some sequences but not `harvard_tea_2`.
13. **Investigate `harvard_tea_2` Warnings**: Listed files and found frame numbers were offset between depth and image files.
14. **Fix Visualization Image Matching (Attempt 2)**: Updated `src/pipelineA/data_processing/dataset.py` to match files by sorted order if file counts match.
15. **Confirm Dataset Inconsistency**: Re-running visualization revealed a mismatch in file counts (33 depth vs 24 image) for `harvard_tea_2`, preventing order-based matching. Concluded remaining warnings for `harvard_tea_2` are due to this dataset inconsistency and cannot be fixed via code logic.

_(Previous changes retained below)_

1.  **Memory Bank Refresh**: Read all core memory bank files to establish context.
2.  **Plan Regularization Experiments**: Confirmed plan to start with `dropout=0.5`.
3.  **Configuration Update (Exp 1)**: Modified `src/pipelineA/config.py` to set `MODEL_PARAMS['dropout'] = 0.5`.
4.  **Execute Training (Exp 1)**: Ran `src/pipelineA/training/train.py` with the updated config. Best validation F1: 0.9552, Acc: 0.9375 at Epoch 45. Run ID: `dgcnn_20250405_145031`.
5.  **Execute Evaluation (Exp 1)**: Ran `src/pipelineA/training/evaluate.py` on the best checkpoint (`model_best.pt`) from Exp 1 against Test Set 1. Results: Acc: 0.8000, Precision: 0.9062, Recall: 0.8056, F1: 0.8529, AUC: 0.8214.

_(Previous changes regarding data split implementation are retained below)_
Work has focused on implementing the new data split strategy:

1.  **Decision**: Agreed to split the Harvard dataset (98 frames) into a validation set (48 frames) and a test set (50 frames) using stratified random sampling to maintain class balance while preserving an unseen test set.
2.  **Label Extraction**: Created and executed `scripts/extract_harvard_labels.py` to determine the binary (Table/No Table) label for each Harvard frame, saving the results to `scripts/harvard_frame_labels.pkl`. Resolved initial Python environment issues (`torch` not found) by ensuring the correct environment was activated.
3.  **Data Splitting**: Created and executed `scripts/split_harvard_data.py` to load the extracted labels and perform the stratified 48/50 split, saving the resulting validation and test frame identifier lists to `scripts/validation_frames.pkl` and `scripts/test_frames.pkl`.
4.  **Configuration Update**: Modified `src/pipelineA/config.py` to load the `VALIDATION_FRAMES` and `TEST_FRAMES` lists from the generated pickle files, replacing the previous hardcoded `TEST1_SEQUENCES`.
5.  **Dataset Loading Update**: Modified `src/pipelineA/data_processing/dataset.py`:
    - Updated `TableDataset.__init__` to accept a `data_spec` (dict for sequences or list for frame IDs).
    - Updated `TableDataset._load_and_filter_samples` to load all potential samples from relevant sequences and then filter based on `data_spec` (using frame IDs for validation/test lists).
    - Updated `create_data_loaders` to use the new `data_spec` argument and load data specifications (MIT sequences, validation frame list, test frame list) from the updated `config.py`.
6.  **Memory Bank Update (Partial)**: Updated `projectbrief.md`, `productContext.md`, `systemPatterns.md`, and `techContext.md` to reflect the new data split strategy and incorporate dataset notes/deliverable requirements from `CW2.pdf`.
7.  **Evaluation Script Alignment**: Modified `src/pipelineA/training/evaluate.py` to instantiate the model using `emb_dims` and `feature_dropout` parameters loaded from `MODEL_PARAMS` in `config.py`, ensuring consistency with the training script. Removed the redundant `--emb_dims` command-line argument.
8.  **Baseline Training**: Completed the baseline training run (`weights/pipelineA/dgcnn_20250405_124525`).
9.  **Baseline Evaluation**: Successfully evaluated the baseline model on Test Set 1 (Harvard-Subset2) using the aligned `evaluate.py`. Results: Acc: 0.7400, Precision: 0.9259, Recall: 0.6944, F1: 0.7937, AUC: 0.8175.
10. **Memory Bank Update (Progress)**: Updated `progress.md` to reflect the completed baseline evaluation and observed overfitting.

## Next Steps

_(Scope revised to focus solely on Pipeline A)_

1.  **Draft Report Sections**: Begin drafting the content for `PipelineA_Report.md` section by section, adhering to the agreed plan:
    - Focus only on Pipeline A.
    - Maintain standard report structure (Intro, Data Processing, Method, Results, Discussion, Conclusion, References).
    - Target approx. 2 pages total length + references.
    - Use Memory Bank content for accuracy without explicit citation within the report.
2.  **Update Memory Bank (Progress)**: Update `progress.md` to reflect the shift to report writing.

## Active Decisions and Considerations

1.  **Dataset Split Strategy (Final)**:
    - **Final Decision**: Use the original split: MIT=Train (290), Harvard-Subset1=Validation (48), Harvard-Subset2=Test1 (50).
    - **Rationale**: The attempt to mitigate domain shift failed. The original split allows for evaluating generalization from MIT to Harvard, which is likely the coursework intent.
2.  **Final Model Selection**:
    - **Final Decision**: Use model checkpoint `dgcnn_20250407_174719` (DGCNN, D=0.5, trained on MIT only - re-run of Exp 1) as the final model for Pipeline A.
    - **Rationale**: This represents the definitive run of the best configuration identified (Exp 1). Confirmed evaluation results (Acc: 0.72, F1: 0.80, AUC: 0.73) show moderate performance but highlight the domain shift limitation (F1=0.0 on `harvard_tea_2`).
3.  **Regularization**:
    - Baseline configuration used augmentation but minimal other regularization (dropout=0, WD=0, clipping=0).
    - **Decision**: Baseline evaluation confirmed overfitting (Val Acc 0.85 vs Test Acc 0.74).
    - **Experiment 1 (DGCNN, D=0.5)**: Identified as best configuration. Re-run (`dgcnn_20250407_174719`) yielded definitive results (Test Acc: 0.72, F1: 0.80).
    - **Experiment 2 (DGCNN, D=0.5, WD=1e-4)**: Worse than Exp 1.
    - **Experiment 3 (DGCNN, D=0.5, FD=0.2)**: Severely hindered learning.
    - **Experiment 4 (DGCNN, D=0.3)**: Slightly worse than Exp 1.
    - **Experiment 5 (PointNet, D=0.5)**: Performed poorly.
    - **Decision**: Confirmed DGCNN with D=0.5 as the best configuration among those tested.
    - **Domain Adaptation Training**: Executed training run `dgcnn_20250407_171414` on mixed data. Evaluation showed poor generalization. Class balance analysis ruled out imbalance as the primary cause. Attempt abandoned.

## Important Patterns and Preferences

1.  **Data Handling**: Prefer loading specific data splits (train/val/test) based on configuration rather than performing splits dynamically within the training script. Use helper scripts for one-off data processing tasks like label extraction and splitting.
2.  **Configuration**: Centralize dataset specifications (sequences, frame lists) and hyperparameters in `config.py`.
3.  **Testing**: Maintain a clear separation between the validation set (used during development) and the test set (used only for final evaluation).
4.  **Script Consistency**: Maintain consistency between training and evaluation scripts, especially regarding model architecture instantiation, by referencing shared configuration files (`config.py`) where possible.
5.  **Custom Data Integration**: Modifications to `config.py`, `dataset.py`, `evaluate.py`, and `visualize_test_predictions.py` allow for evaluating the custom UCL dataset (now designated Test Set 2) using `EVAL_TEST_SET=2`. Requires the dataset to follow the expected directory structure (depth/, image/, intrinsics.txt) and have a text label file specified in `UCL_DATA_CONFIG`.
6.  **Configuration Centralization**: Scripts (`train.py`, `evaluate.py`, `visualize_test_predictions.py`) now exclusively use `config.py` for all configuration parameters, removing the need for command-line arguments and ensuring consistency.
7.  **Script Standardization**: Aligning `visualize_test_predictions.py` with `evaluate.py` improves maintainability and reduces potential inconsistencies by using shared utilities (`get_model`, `load_checkpoint`), configuration (`config.py`), and dataset selection logic (now unified for Test Set 1 and 2).

## Learnings and Project Insights

1.  **Test Set Integrity**: Reaffirmed the importance of having a truly unseen test set for unbiased evaluation. Using validation data for final testing leads to inflated results.
2.  **Data Splitting**: Stratified sampling is crucial when splitting smaller datasets to ensure class distributions are preserved across subsets, especially for classification tasks.
3.  **Code Modularity**: Refactoring `dataset.py` to handle different types of data specifications (sequence dicts vs frame lists) improves flexibility.
4.  **Overfitting Confirmed**: The baseline run demonstrated overfitting (Val Acc 0.85 vs Test Acc 0.74).
5.  **Dropout Impact**: Experiment 1 showed `dropout=0.5` significantly improves overall performance but doesn't resolve the Val-Test gap.
6.  **Weight Decay Impact**: Experiment 2 showed `weight_decay=1e-4` was detrimental when combined with `dropout=0.5`.
7.  **Feature Dropout Impact**: Experiment 3 showed `feature_dropout=0.2` combined with `dropout=0.5` was highly detrimental.
8.  **Dropout Rate**: `dropout=0.5` was more effective than `0.3` for DGCNN.
9.  **Model Architecture**: DGCNN significantly outperformed PointNet on this task/dataset with the current setup.
10. **Overfitting Analysis (Val vs Test1)**: Completed analysis using checkpoint `dgcnn_20250407_142213/model_best.pt`. Key findings:
    - Confirmed performance drop (Acc: 0.875 -> 0.800, F1: 0.914 -> 0.865).
    - Significant drop in AUC-ROC (0.939 -> 0.794), indicating poorer class discrimination on unseen data.
    - Recall drop (0.941 -> 0.889) suggests more tables are missed on the test set.
    - Model struggles with negative samples (`harvard_tea_2`), yielding F1=0.0 on both sets. This sequence is now part of the training data.
    - Qualitative review of visualizations is needed for deeper insight into error types.
11. **Domain Shift Impact**: The difference between DepthTSDF and raw depth is confirmed as a significant factor affecting generalization. The attempt to mitigate this by training on mixed data failed. This will be a key point for discussion in the final report.
