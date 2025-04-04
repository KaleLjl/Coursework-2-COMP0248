# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The primary focus is now on **investigating the cause of the high, initially flat validation F1 score** (observed around 0.83 in early epochs). The leading hypothesis is that this score reflects the model initially predicting the majority class ("no table") in an imbalanced Harvard validation set.

Key areas:
1.  **Validation Set Analysis**: Confirming the class imbalance in the Harvard dataset.
2.  **Code Review**: Examining evaluation logic (`evaluate.py`), training loop (`train.py`), and model initialization (`classifier.py`) for factors contributing to this initial behavior.
3.  **Training with Optimal Diagnostic Config**: Once the initial score behavior is understood, the plan remains to run a full training session using the 'augmentation only' configuration (augmentation enabled, dropout=0.0, weight_decay=0.0, gradient_clip=0.0) identified as promising in diagnostics.
4.  **Performance Evaluation**: Analyzing the results of the 'augmentation only' run, monitoring for overfitting, and comparing against the initial baseline behavior.

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
6.  **Depth Warning Debugging & `harvard_tea_2` Handling**:
    *   Identified "No valid depth values" warnings during training, specifically for `harvard_tea_2` sequence.
    *   Enhanced logging in `depth_to_pointcloud.py` to print full paths and raw depth statistics.
    *   Confirmed the issue was caused by the `max_depth` threshold (10.0m) being too low for the millimeter-based raw depth values in `harvard_tea_2` after conversion to meters.
    *   Fixed by increasing `POINT_CLOUD_PARAMS['max_depth']` to `20.0` in `config.py`.
    *   Corrected an `IndentationError` introduced during debugging in `depth_to_pointcloud.py`.
    *   Confirmed `dataset.py` uses a `use_raw_depth` flag to correctly select the `depth` directory (vs `depthTSDF`) and pass this flag to `create_pointcloud_from_depth` for appropriate scaling (likely mm to m conversion).
7.  **Validation Data Shuffling**:
    *   Modified `dataset.py`: Updated `create_data_loaders` to set `shuffle=True` for the `val_loader` as requested.

## Next Steps

Priority is now investigating the initial high validation score:

1.  **Update Memory Bank**: Document recent changes, resolved environment issue, validation score investigation priority, and updated plan. (In Progress).
2.  **Calculate Validation Set Distribution**: Implement logic (potentially a temporary script or modification in `dataset.py`) to count the number of "table" vs "no table" samples in the Harvard validation set loaded by `val_dataset`.
3.  **Review Evaluation Logic (`evaluate.py`)**: Check metric calculations (Precision, Recall, F1) for correctness, especially regarding averaging methods (`binary`, `micro`, `macro`, `weighted`) and handling of zero divisions in the context of potential imbalance.
4.  **Review Training Logic (`train.py`)**: Confirm `model.eval()` is correctly used before validation. Check loss function setup.
5.  **Review Model Initialization (`classifier.py`)**: Briefly check weight initialization.
6.  **Set 'Augmentation Only' Configuration**: Once the initial score behavior is understood, ensure `config.py` reflects:
    *   `MODEL_PARAMS['dropout'] = 0.0`
    *   `MODEL_PARAMS['feature_dropout'] = 0.0`
    *   `TRAIN_PARAMS['weight_decay'] = 0.0`
    *   `TRAIN_PARAMS['gradient_clip'] = 0.0`
    *   `AUGMENTATION_PARAMS['enabled'] = True`
    *   `TRAIN_PARAMS['num_epochs'] = 100` # Or adjust based on findings
7.  **Run Training**: Execute `train.py` with the 'augmentation only' configuration.
8.  **Analyze Performance**: Monitor logs, evaluate peak performance, check for overfitting, compare against initial behavior.
9.  **Iterate (If Needed)**: If performance is good but overfitting occurs, consider reintroducing mild regularization.
10. **Memory Bank Update**: Update `progress.md` with investigation findings and training results.

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
   - Point cloud data presents unique generalization challenges compared to images.
   - Domain-specific augmentations are crucial for improving generalization.
   - Sequence-specific patterns may be learned instead of general table characteristics.
   - The true test of generalization will now be performance on the Harvard validation sequences.
5. **Flat Validation Metrics Diagnosis**:
   - Diagnostic tests confirmed that the high dropout rates (`dropout=0.7`, `feature_dropout=0.2`) used previously were the primary cause of the flat validation metrics.
   - Configurations without dropout, even with augmentation and other regularization (weight decay, gradient clipping), showed dynamic validation performance.
   - This indicates the model was overly constrained by the high dropout, preventing effective learning on the validation set distribution.
6. **Baseline Run Insights & Diagnostic Summary**:
   - Diagnostic tests isolated high dropout (`0.7`/`0.2`) as the cause of flat validation metrics.
   - A diagnostic run with only augmentation enabled achieved the highest peak F1 (0.9306) compared to runs with weight decay/clipping enabled (peak F1=0.8333).
   - This suggests the best path forward is to start with the minimal 'augmentation only' configuration and potentially add back *mild* regularization later if overfitting becomes an issue.
7. **Initial F1 Score Insight**:
   - The consistent starting validation F1 score of 0.8333 (with accuracy 0.7143) strongly suggests the model initially defaults to predicting the majority class ("no table"), and 0.8333 is the F1 score for that majority class given the likely 71.4% prevalence in the validation set.
   - The unused `mixup_alpha` parameter remains a potential tool if needed later.
8. **Environment Stability**: The previous `ModuleNotFoundError` was confirmed by the user to be related to environment activation and is considered resolved.
9. **Validation Data Loading**: The validation loader now shuffles data (`shuffle=True` in `dataset.py`), matching the training loader behavior. This might slightly change epoch-to-epoch validation scores compared to non-shuffled evaluation, but the overall trend should be similar.
