# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The primary focus is now on **concluding the Pipeline A improvement phase** and **planning the next major phase** of the project. This involves:
1.  **Result Analysis (Exp 5 - PointNet)**: Reviewing the performance metrics (Val Acc 0.7083, Test Acc 0.7200, F1 0.8372, AUC 0.4226). PointNet performed poorly, essentially classifying everything as Table.
2.  **Conclusion**: Pipeline A tuning complete for now. Experiment 1 (DGCNN, D=0.5) remains the best configuration (Test Acc 0.8000, F1 0.8529). Switching to PointNet was detrimental.
3.  **Memory Bank Update**: Documenting Experiment 5 results and the conclusion.
4.  **Next Phase Planning**: Recommend starting Pipeline B or C.

## Recent Changes

1.  **Update Memory Bank (Post-Exp 4)**: Documented Experiment 4 results in `activeContext.md` and `progress.md`.
2.  **Configuration Update (Exp 5)**: Modified `src/pipelineA/config.py` to set `MODEL_PARAMS['model_type'] = 'pointnet'` (keeping D=0.5, WD=0, FD=0).
3.  **Execute Training (Exp 5 - PointNet)**: Ran `src/pipelineA/training/train.py --model pointnet`. Best validation F1: 0.8293, Acc: 0.7083 at Epoch 2. Run ID: `pointnet_20250405_155003`.
4.  **Execute Evaluation (Exp 5 - PointNet)**: Ran `src/pipelineA/training/evaluate.py --model_type pointnet` on the best checkpoint. Results: Acc: 0.7200, Precision: 0.7200, Recall: 1.0000, F1: 0.8372, AUC: 0.4226. Poor performance.

*(Previous changes retained below)*
1.  **Memory Bank Refresh**: Read all core memory bank files to establish context.
2.  **Plan Regularization Experiments**: Confirmed plan to start with `dropout=0.5`.
3.  **Configuration Update (Exp 1)**: Modified `src/pipelineA/config.py` to set `MODEL_PARAMS['dropout'] = 0.5`.
4.  **Execute Training (Exp 1)**: Ran `src/pipelineA/training/train.py` with the updated config. Best validation F1: 0.9552, Acc: 0.9375 at Epoch 45. Run ID: `dgcnn_20250405_145031`.
5.  **Execute Evaluation (Exp 1)**: Ran `src/pipelineA/training/evaluate.py` on the best checkpoint (`model_best.pt`) from Exp 1 against Test Set 1. Results: Acc: 0.8000, Precision: 0.9062, Recall: 0.8056, F1: 0.8529, AUC: 0.8214.

*(Previous changes regarding data split implementation are retained below)*
Work has focused on implementing the new data split strategy:

1.  **Decision**: Agreed to split the Harvard dataset (98 frames) into a validation set (48 frames) and a test set (50 frames) using stratified random sampling to maintain class balance while preserving an unseen test set.
2.  **Label Extraction**: Created and executed `scripts/extract_harvard_labels.py` to determine the binary (Table/No Table) label for each Harvard frame, saving the results to `scripts/harvard_frame_labels.pkl`. Resolved initial Python environment issues (`torch` not found) by ensuring the correct environment was activated.
3.  **Data Splitting**: Created and executed `scripts/split_harvard_data.py` to load the extracted labels and perform the stratified 48/50 split, saving the resulting validation and test frame identifier lists to `scripts/validation_frames.pkl` and `scripts/test_frames.pkl`.
4.  **Configuration Update**: Modified `src/pipelineA/config.py` to load the `VALIDATION_FRAMES` and `TEST_FRAMES` lists from the generated pickle files, replacing the previous hardcoded `TEST1_SEQUENCES`.
5.  **Dataset Loading Update**: Modified `src/pipelineA/data_processing/dataset.py`:
    *   Updated `TableDataset.__init__` to accept a `data_spec` (dict for sequences or list for frame IDs).
    *   Updated `TableDataset._load_and_filter_samples` to load all potential samples from relevant sequences and then filter based on `data_spec` (using frame IDs for validation/test lists).
    *   Updated `create_data_loaders` to use the new `data_spec` argument and load data specifications (MIT sequences, validation frame list, test frame list) from the updated `config.py`.
6.  **Memory Bank Update (Partial)**: Updated `projectbrief.md`, `productContext.md`, `systemPatterns.md`, and `techContext.md` to reflect the new data split strategy and incorporate dataset notes/deliverable requirements from `CW2.pdf`.
7.  **Evaluation Script Alignment**: Modified `src/pipelineA/training/evaluate.py` to instantiate the model using `emb_dims` and `feature_dropout` parameters loaded from `MODEL_PARAMS` in `config.py`, ensuring consistency with the training script. Removed the redundant `--emb_dims` command-line argument.
8.  **Baseline Training**: Completed the baseline training run (`weights/pipelineA/dgcnn_20250405_124525`).
9.  **Baseline Evaluation**: Successfully evaluated the baseline model on Test Set 1 (Harvard-Subset2) using the aligned `evaluate.py`. Results: Acc: 0.7400, Precision: 0.9259, Recall: 0.6944, F1: 0.7937, AUC: 0.8175.
10. **Memory Bank Update (Progress)**: Updated `progress.md` to reflect the completed baseline evaluation and observed overfitting.

## Next Steps

1.  **Update Memory Bank**: Document Experiment 5 results and conclusion in `activeContext.md` (this update) and `progress.md`.
2.  **Revert Config**: Modify `src/pipelineA/config.py` to set `MODEL_PARAMS['model_type'] = 'dgcnn'` (reverting from PointNet).
3.  **Propose Next Phase**: Ask user to choose between starting Pipeline B or Pipeline C.
4.  **(Lower Priority)** RealSense data collection.

## Active Decisions and Considerations

1.  **Dataset Split Strategy**:
    *   **Decision**: Implement MIT=Train (290), Harvard-Subset1=Validation (48), Harvard-Subset2=Test1 (50) using stratified random sampling.
    *   **Rationale**: Provides a mechanism for monitoring generalization and guiding training (validation set) while maintaining a truly unseen test set for final evaluation, addressing methodological concerns of the previous approach.
2.  **Regularization**:
    *   Baseline configuration used augmentation but minimal other regularization (dropout=0, WD=0, clipping=0).
    *   **Decision**: Baseline evaluation confirmed overfitting (Val Acc 0.85 vs Test Acc 0.74).
    *   **Experiment 1 (DGCNN, D=0.5)**: Best result so far (Test Acc 0.8000, F1 0.8529).
    *   **Experiment 2 (DGCNN, D=0.5, WD=1e-4)**: Worse than Exp 1.
    *   **Experiment 3 (DGCNN, D=0.5, FD=0.2)**: Severely hindered learning.
    *   **Experiment 4 (DGCNN, D=0.3)**: Slightly worse than Exp 1.
    *   **Experiment 5 (PointNet, D=0.5)**: Performed poorly (Test Acc 0.7200, AUC ~0.42).
    *   **Decision**: Conclude Pipeline A tuning/model exploration for now. DGCNN with D=0.5 is the best configuration. Revert config. Proceed to next pipeline implementation.

## Important Patterns and Preferences

1.  **Data Handling**: Prefer loading specific data splits (train/val/test) based on configuration rather than performing splits dynamically within the training script. Use helper scripts for one-off data processing tasks like label extraction and splitting.
2.  **Configuration**: Centralize dataset specifications (sequences, frame lists) and hyperparameters in `config.py`.
3.  **Testing**: Maintain a clear separation between the validation set (used during development) and the test set (used only for final evaluation).
4.  **Script Consistency**: Maintain consistency between training and evaluation scripts, especially regarding model architecture instantiation, by referencing shared configuration files (`config.py`) where possible.

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
