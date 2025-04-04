# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The project is currently focused on the following areas:

1. **Pipeline A Improvement**: The depth-to-point-cloud-to-classification pipeline is being restructured with a major dataset split change to improve generalization performance.

2. **Dataset Split Restructuring**: The dataset split has been completely revised: MIT sequences (larger dataset) will be used for training, Harvard sequences (smaller dataset) for validation, and the test dataset will remain empty for now. This should provide a stronger validation signal and better measure of generalization.

3. **Model Overfitting Investigation**: In-depth analysis confirms the model has high capacity relative to the limited training data, allowing it to memorize training examples rather than learn generalizable features. The new dataset split should help address this issue.

## Recent Changes

Recent work has focused on implementing the strategy to address overfitting:

1.  **Label Format Handling Confirmed**: Reviewed `dataset.py` and confirmed existing logic correctly handles the "depth" vs "depthTSDF" format difference for `harvard_tea_2`. No code changes were needed for this specific item.

2.  **New Dataset Split Strategy Implemented**:
    *   Modified `dataset.py`: Updated `create_data_loaders` to accept separate `train_sequences` and `val_sequences` arguments, removing the old `train_val_split` logic.
    *   Modified `train.py`: Updated the call to `create_data_loaders` to pass `TRAIN_SEQUENCES` (MIT) as training data and `TEST1_SEQUENCES` (Harvard) as validation data.

3.  **Enhanced Model Regularization Implemented**:
    *   Modified `classifier.py`: Added `feature_dropout` parameter to `DGCNN` constructor and applied dropout after EdgeConv layers during training. Updated `get_model` to accept this parameter.
    *   Confirmed `config.py`: Verified that `MODEL_PARAMS` already contained `emb_dims` and `feature_dropout` parameters.

4.  **Advanced Data Augmentation Implemented**:
    *   Modified `preprocessing.py`: Added `point_dropout` and `random_subsample` functions. Integrated these into the `augment_point_cloud` function and updated the call within `preprocess_point_cloud`.

5.  **Training Process Enhancements Implemented**:
    *   Modified `train.py`:
        *   Updated model instantiation to use regularization parameters (`emb_dims`, `dropout`, `feature_dropout`) from `config.py`.
        *   Updated optimizer and scheduler instantiation to use parameters (`weight_decay`, `lr_scheduler_factor`, `lr_scheduler_patience`) from `config.py`.
        *   Implemented gradient clipping in `train_epoch` using `TRAIN_PARAMS['gradient_clip']`.
        *   Simplified command-line arguments, removing those now sourced from `config.py`.
6.  **Depth Warning Debugging**:
    *   Identified "No valid depth values" warnings during training, specifically for `harvard_tea_2` sequence.
    *   Enhanced logging in `depth_to_pointcloud.py` to print full paths and raw depth statistics.
    *   Confirmed the issue was caused by the `max_depth` threshold (10.0m) being too low for the millimeter-based raw depth values in `harvard_tea_2` after conversion to meters.
    *   Fixed by increasing `POINT_CLOUD_PARAMS['max_depth']` to `20.0` in `config.py`.
    *   Corrected an `IndentationError` introduced during debugging in `depth_to_pointcloud.py`.

## Next Steps

With the core code changes for the overfitting mitigation strategy complete and the depth warning issue resolved, the immediate next steps are:

1.  **Testing and Validation**:
    *   Re-run the updated training script (`train.py`) with the corrected configuration (MIT train, Harvard val, `max_depth=20.0`, enhanced regularization/augmentation).
    *   Monitor training progress using TensorBoard, paying close attention to the validation metrics (especially F1-score) on the Harvard set and the train/validation divergence. Ensure the depth warnings are gone.
    *   Evaluate the performance of the best model checkpoint on the Harvard validation set.
    *   Compare generalization performance (e.g., DGCNN vs. PointNet, different `emb_dims`).
    *   Create visualizations to understand model behavior.

2.  **Further Training Enhancements (Optional/Iterative)**:
    *   Based on initial results, consider implementing mixup augmentation (`mixup_alpha` parameter exists in `config.py` but is not yet used in `train.py`).
    *   Implement more detailed monitoring of train/validation divergence.
    *   Experiment with reduced model complexity (e.g., `emb_dims=512` via `config.py`).

3.  **Memory Bank Update**: Update `progress.md` after initial training runs.

## Active Decisions and Considerations

Key decisions currently being made:

1. **Dataset Split Strategy**:
   - Decided to use MIT sequences (~290 frames) for training, Harvard sequences (~98 frames) for validation
   - This addresses the weak validation signal from the previous approach and should better measure generalization
   - Using the larger dataset for training should help with model learning

2. **Model Architecture Strategy**: 
   - Confirmed that the model has too much capacity relative to the dataset size
   - Need to properly balance model complexity for the training dataset
   - Focus on techniques that improve generalization to the Harvard validation set

3. **Advanced Regularization Techniques**:
   - Dropout should be increased beyond standard values (0.5 → 0.7)
   - Weight decay needs to be more aggressive (1e-4 → 5e-4)
   - Feature-level dropout and gradient clipping should be implemented
   - Regularization is key to ensuring the model doesn't overfit the MIT training set

4. **Testing and Evaluation Focus**:
   - Harvard validation set will now be the primary indicator of generalization performance
   - Need to establish new baselines with this dataset configuration
   - Implementing more robust monitoring of training/validation divergence will be critical

## Important Patterns and Preferences

1. **Regularization Strategy**: 
   - Confirmed GroupNorm and LayerNorm are good choices for varying batch sizes
   - Multiple dropout strategies should be combined: standard dropout, feature dropout, and input dropout
   - Weight decay should be aggressively tuned based on validation performance

2. **Data Augmentation Best Practices**:
   - Point cloud augmentation needs to be more aggressive than initially implemented
   - Combining multiple augmentation types is critical: rotation, jitter, scaling, point dropout
   - Random subsampling during training can help prevent overfitting to specific point densities

3. **Training Protocol**:
   - Continue using TensorBoard for visualizing metrics
   - Add new metrics focusing specifically on overfitting detection
   - Maintain early stopping based on validation F1-score

4. **Model Selection Strategy**:
   - DGCNN remains the primary architecture but with enhanced regularization
   - PointNet should be evaluated as a potentially more generalizable alternative
   - Consider ensemble methods only after optimizing individual models

## Learnings and Project Insights

Key insights gained from investigating the overfitting issue:

1. **Dataset Structure Understanding**:
   - The train/validation split strategy is critical - previous random frame-level splitting from MIT sequences did not adequately test generalization
   - MIT sequences and Harvard sequences likely have distribution differences that make cross-dataset generalization challenging
   - Using MIT sequences (larger dataset) for training and Harvard sequences (smaller dataset) for validation provides a better approach
   - The model needs to learn from more diverse examples before being tested on a different dataset

2. **Model Architecture Insights**:
   - DGCNN's edge convolution operations can easily overfit to training set patterns
   - The embedding dimension (1024) is likely too large for the dataset size
   - Regularization needs to be applied at multiple levels: weights, features, and input data
   - Need to find model architectures that generalize well from MIT to Harvard data

3. **Training Dynamics Observations**:
   - The divergence between training and validation performance begins after ~20-30 epochs
   - There is indeed a critical early phase where generalization is learned before memorization
   - Learning rate scheduling and early stopping are important but insufficient alone
   - With the revised dataset split, we expect to see new training dynamics that will need careful monitoring

4. **Generalization Challenges**:
   - Point cloud data presents unique generalization challenges compared to images
   - Domain-specific augmentations are crucial for improving generalization
   - Sequence-specific patterns may be learned instead of general table characteristics
   - The true test of generalization will now be performance on the Harvard validation sequences
