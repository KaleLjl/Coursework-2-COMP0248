# Active Context: Table Detection from 3D Point Clouds

## Current Focus

The project is currently focused on the following areas:

1. **Pipeline A Improvement**: The depth-to-point-cloud-to-classification pipeline is experiencing an issue where training metrics (accuracy and F1-score) are increasing, but validation metrics are not improving correspondingly.

2. **Model Overfitting Investigation**: In-depth analysis confirms this is a classic overfitting problem. The model has high capacity relative to the limited training data, allowing it to memorize training examples rather than learn generalizable features.

3. **Dataset Split Analysis**: The current train/validation split uses a random 80/20 division of frames from the same MIT sequences, which means validation samples may be very similar to training samples. This may not fully test generalization capabilities.

## Recent Changes

Recent work has focused on analyzing and addressing the overfitting issue:

1. **Code Analysis**: Thorough review of all model, dataset, and training code to identify overfitting causes.
2. **Configuration Updates**: Enhanced regularization and data augmentation parameters in `config.py`:
   - Increased dropout rate from 0.5 to 0.7
   - Added feature-level dropout (0.2)
   - Increased weight decay from 1e-4 to 5e-4
   - Added gradient clipping
   - Enhanced data augmentation with more aggressive rotations, jitter, and point dropout
3. **Dataset Split Understanding**: Identified that train and validation data come from the same MIT sequences split randomly at the frame level, while true test data comes from Harvard sequences.

## Next Steps

The immediate next steps for addressing the overfitting problem are:

1. **Implement Enhanced Model Regularization**:
   - Modify DGCNN model to support feature-level dropout
   - Test reduced model complexity by decreasing embedding dimensions (1024 → 512)
   - Add spectral normalization to convolutional layers

2. **Improve Dataset Splitting Strategy**:
   - Implement sequence-based train/validation split instead of random frame-based split
   - Create validation set from distinct MIT sequences to better test generalization
   - Consider cross-sequence validation to ensure robust evaluation

3. **Enhance Training Process**:
   - Implement mixup augmentation for point clouds
   - Add monitoring of train/validation divergence with early warnings
   - Test gradient accumulation for more stable optimization

4. **Advanced Data Augmentation**:
   - Add point dropout implementation to simulate occlusion
   - Implement partial point cloud rotation to create more viewpoint variety
   - Test random subsampling during training

5. **Testing and Validation**:
   - Evaluate model performance on Harvard test sequences
   - Compare generalization of DGCNN vs. PointNet architectures
   - Create visualizations to understand what features the model is learning

## Active Decisions and Considerations

Key decisions currently being made:

1. **Model Architecture Strategy**: 
   - Confirmed that the model has too much capacity relative to the dataset size (~290 frames)
   - Implementing stronger regularization as the primary approach, while also testing reduced model complexity
   - Adding more aggressive data augmentation to artificially increase effective dataset size

2. **Dataset Split Strategy**:
   - The current random frame-level split may not properly test generalization
   - Considering sequence-level splitting to provide a stronger validation signal
   - Evaluating whether to use a portion of the Harvard data for validation

3. **Advanced Regularization Techniques**:
   - Dropout should be increased beyond standard values (0.5 → 0.7)
   - Weight decay needs to be more aggressive (1e-4 → 5e-4)
   - Feature-level dropout and gradient clipping should be implemented

4. **Testing and Evaluation Focus**:
   - Need to prioritize validation performance over training performance
   - Ensuring that validation metrics truly measure generalization capability
   - Implementing more robust monitoring of training/validation divergence

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
   - The train/validation split strategy is critical - random frame-level splitting may not adequately test generalization
   - MIT sequences (training) and Harvard sequences (testing) likely have distribution differences that must be addressed
   - The limited dataset size (~290 frames) makes overfitting a significant risk with high-capacity models

2. **Model Architecture Insights**:
   - DGCNN's edge convolution operations can easily overfit to training set patterns
   - The embedding dimension (1024) is likely too large for the dataset size
   - Regularization needs to be applied at multiple levels: weights, features, and input data

3. **Training Dynamics Observations**:
   - The divergence between training and validation performance begins after ~20-30 epochs
   - There is indeed a critical early phase where generalization is learned before memorization
   - Learning rate scheduling and early stopping are important but insufficient alone

4. **Generalization Challenges**:
   - Point cloud data presents unique generalization challenges compared to images
   - Domain-specific augmentations are crucial for improving generalization
   - Sequence-specific patterns may be learned instead of general table characteristics
   - The true test of generalization will be performance on the Harvard sequences
