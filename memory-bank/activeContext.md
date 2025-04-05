# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The primary focus is now on **concluding the initial regularization tuning for Pipeline A** and **planning the next major phase** of the project. This involves:
1.  **Result Analysis (Exp 4)**: Reviewing the performance metrics (Val Acc 0.9167, Test Acc 0.7800, F1 0.8533). Note that D=0.3 performed slightly worse than D=0.5 (Exp 1).
2.  **Conclusion**: Experiment 1 (D=0.5 only) remains the best configuration (Test Acc 0.8000, F1 0.8529), despite persistent overfitting (Val-Test gap ~0.14). Further tuning of these specific parameters is unlikely to yield major gains.
3.  **Memory Bank Update**: Documenting Experiment 4 results and the conclusion.
4.  **Next Phase Planning**: Discuss options: Implement Pipeline B, Implement Pipeline C, explore other Pipeline A improvements (e.g., different model, augmentations).

## Recent Changes

1.  **Update Memory Bank (Post-Exp 3)**: Documented Experiment 3 results in `activeContext.md` and `progress.md`.
2.  **Configuration Update (Exp 4)**: Modified `src/pipelineA/config.py` to set `MODEL_PARAMS['feature_dropout'] = 0.0` and `MODEL_PARAMS['dropout'] = 0.3` (keeping `weight_decay=0.0`).
3.  **Execute Training (Exp 4)**: Ran `src/pipelineA/training/train.py` with the updated config. Best validation F1: 0.9167, Acc: 0.9167 at Epoch 37. Run ID: `dgcnn_20250405_152915`.
4.  **Execute Evaluation (Exp 4)**: Ran `src/pipelineA/training/evaluate.py` on the best checkpoint (`model_best.pt`) from Exp 4 against Test Set 1. Results: Acc: 0.7800, Precision: 0.8205, Recall: 0.8889, F1: 0.8533, AUC: 0.8313. Slightly worse than Exp 1.

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

1.  **Update Memory Bank**: Document Experiment 4 results and conclusion in `activeContext.md` (this update) and `progress.md`.
2.  **Revert Config**: Modify `src/pipelineA/config.py` to set `MODEL_PARAMS['dropout'] = 0.5` (the best performing setting from Exp 1).
3.  **Discuss Next Phase**: Engage with the user to decide whether to start Pipeline B, Pipeline C, or explore other improvements for Pipeline A.
4.  **(Lower Priority)** RealSense data collection.

## Active Decisions and Considerations

1.  **Dataset Split Strategy**:
    *   **Decision**: Implement MIT=Train (290), Harvard-Subset1=Validation (48), Harvard-Subset2=Test1 (50) using stratified random sampling.
    *   **Rationale**: Provides a mechanism for monitoring generalization and guiding training (validation set) while maintaining a truly unseen test set for final evaluation, addressing methodological concerns of the previous approach.
2.  **Regularization**:
    *   Baseline configuration used augmentation but minimal other regularization (dropout=0, WD=0, clipping=0).
    *   **Decision**: Baseline evaluation confirmed overfitting (Val Acc 0.85 vs Test Acc 0.74).
    *   **Experiment 1 (Dropout=0.5)**: Improved Val (0.9375) and Test (0.8000) accuracy significantly, but the Val-Test gap slightly increased (0.1375). **Best result so far.**
    *   **Experiment 2 (Dropout=0.5, WD=1e-4)**: Decreased performance compared to Exp 1 (Test Acc 0.7600). Val-Test gap remained similar (0.1358).
    *   **Experiment 3 (Dropout=0.5, FD=0.2)**: Severely hindered learning (Test Acc 0.7200, AUC ~0.5).
    *   **Experiment 4 (Dropout=0.3)**: Performed slightly worse than Exp 1 (Test Acc 0.7800). Val-Test gap similar (0.1367).
    *   **Decision**: Conclude initial regularization tuning. Experiment 1 (D=0.5 only) is the best configuration for Pipeline A currently. Revert config to D=0.5. Plan next major project phase.

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
8.  **Dropout Rate**: Experiment 4 showed `dropout=0.3` was slightly less effective than `dropout=0.5` (Exp 1).
