# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The project is currently focused on the following areas:

1. **Pipeline A Improvement**: The depth-to-point-cloud-to-classification pipeline is being restructured with a major dataset split change to improve generalization performance.

2. **Dataset Split Restructuring**: The dataset split has been completely revised: MIT sequences (larger dataset) will be used for training, Harvard sequences (smaller dataset) for validation, and the test dataset will remain empty for now. This should provide a stronger validation signal and better measure of generalization.

3. **Model Overfitting Investigation**: In-depth analysis confirms the model has high capacity relative to the limited training data, allowing it to memorize training examples rather than learn generalizable features. The new dataset split should help address this issue.

## Recent Changes

Recent work has focused on analyzing and addressing the overfitting issue:

1. **Major Dataset Split Restructuring**: Changed the dataset usage:
   - MIT sequences (~290 RGBD frames) will be used for training
   - Harvard sequences (~98 RGBD frames) will be used for validation
   - Test dataset will remain empty for now
   
2. **Configuration Updates**: Enhanced regularization and data augmentation parameters in `config.py`:
   - Increased dropout rate from 0.5 to 0.7
   - Added feature-level dropout (0.2)
   - Increased weight decay from 1e-4 to 5e-4
   - Added gradient clipping
   - Enhanced data augmentation with more aggressive rotations, jitter, and point dropout

3. **Dataset Split Understanding**: Previous approach used train and validation data from the same MIT sequences split randomly at the frame level, while test data came from Harvard sequences. This new approach should provide a much stronger signal about generalization performance.

4. **Label Format Differences**: Identified that the harvard_tea_2 dataset only has the "depth" label format, while other datasets have the "depthTSDF" label format. This requires special handling in the data loading pipeline to ensure consistent processing across all validation sequences.

## Next Steps

The immediate next steps are:

1. **Address Label Format Inconsistency**:
   - Implement special handling for harvard_tea_2 dataset which uses "depth" label format
   - Ensure consistent data processing between different label formats
   - Add format detection and conversion logic in the data loading pipeline

2. **Implement New Dataset Split Strategy**:
   - Update the dataset loading code to use MIT sequences for training
   - Configure Harvard sequences as the validation set
   - Leave test dataset empty for now
   - Update data processing pipelines to handle this new configuration

2. **Implement Enhanced Model Regularization**:
   - Modify DGCNN model to support feature-level dropout
   - Test reduced model complexity by decreasing embedding dimensions (1024 → 512)
   - Add spectral normalization to convolutional layers
   - Adjust regularization parameters appropriately

3. **Enhance Training Process**:
   - Implement mixup augmentation for point clouds
   - Add monitoring of train/validation divergence with early warnings
   - Test gradient accumulation for more stable optimization
   - Focus on generalization to the Harvard validation set

5. **Advanced Data Augmentation**:
   - Add point dropout implementation to simulate occlusion
   - Implement partial point cloud rotation to create more viewpoint variety
   - Test random subsampling during training

6. **Testing and Validation**:
   - Evaluate model performance on Harvard validation sequences
   - Compare generalization of DGCNN vs. PointNet architectures
   - Create visualizations to understand what features the model is learning
   - Ensure harvard_tea_2 sequence is properly evaluated despite label format differences

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
