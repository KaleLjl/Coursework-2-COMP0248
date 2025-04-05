# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The primary focus is on establishing a reliable baseline for Pipeline A using the new stratified data split. This involves:
1.  **Code Consistency**: Ensuring training, evaluation, and configuration scripts are aligned (e.g., recent alignment of `evaluate.py` model instantiation).
2.  **Baseline Training Run**: Executing an initial training run (`train.py`) using the new data split and baseline configuration to establish performance benchmarks.
3.  **Initial Evaluation**: Preparing for the first evaluation run on the held-out test set (`evaluate.py`) after the baseline training completes.

## Recent Changes

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

## Next Steps

1.  **Finalize Memory Bank**: Update `progress.md` to reflect the completed data split implementation, `evaluate.py` alignment, and the current project status.
2.  **Verify Data Loaders**: (Already marked as complete in `progress.md`, but good to keep in mind if issues arise). Confirm `create_data_loaders` correctly loads expected sample counts.
3.  **Run Baseline Training**: Execute `src/pipelineA/training/train.py` using the current configuration (DGCNN, Augmentation=True, Dropout=0, WD=0) to establish baseline performance with the new data split. Monitor training/validation metrics.
4.  **Evaluate Baseline on Test Set**: After the baseline training run completes, use the *now aligned* `src/pipelineA/training/evaluate.py` to evaluate the final trained model on the held-out test set (Harvard-Subset2, Test Set 1).
5.  **Analyze Baseline Results**: Review the baseline performance (train/val/test metrics), check for overfitting/underfitting.
6.  **Plan Further Experiments**: Based on baseline results, decide on next steps for Pipeline A (e.g., hyperparameter tuning, reintroducing regularization) or begin work on Pipeline B/C or RealSense data collection.

## Active Decisions and Considerations

1.  **Dataset Split Strategy**:
    *   **Decision**: Implement MIT=Train (290), Harvard-Subset1=Validation (48), Harvard-Subset2=Test1 (50) using stratified random sampling.
    *   **Rationale**: Provides a mechanism for monitoring generalization and guiding training (validation set) while maintaining a truly unseen test set for final evaluation, addressing methodological concerns of the previous approach.
2.  **Regularization**:
    *   Current configuration uses augmentation but minimal other regularization (dropout=0, WD=0, clipping=0) based on previous diagnostic tests.
    *   This needs re-evaluation after the baseline run with the new data split, as overfitting behavior might change.

## Important Patterns and Preferences

1.  **Data Handling**: Prefer loading specific data splits (train/val/test) based on configuration rather than performing splits dynamically within the training script. Use helper scripts for one-off data processing tasks like label extraction and splitting.
2.  **Configuration**: Centralize dataset specifications (sequences, frame lists) and hyperparameters in `config.py`.
3.  **Testing**: Maintain a clear separation between the validation set (used during development) and the test set (used only for final evaluation).
4.  **Script Consistency**: Maintain consistency between training and evaluation scripts, especially regarding model architecture instantiation, by referencing shared configuration files (`config.py`) where possible.

## Learnings and Project Insights

1.  **Test Set Integrity**: Reaffirmed the importance of having a truly unseen test set for unbiased evaluation. Using validation data for final testing leads to inflated results.
2.  **Data Splitting**: Stratified sampling is crucial when splitting smaller datasets to ensure class distributions are preserved across subsets, especially for classification tasks.
3.  **Code Modularity**: Refactoring `dataset.py` to handle different types of data specifications (sequence dicts vs frame lists) improves flexibility.
