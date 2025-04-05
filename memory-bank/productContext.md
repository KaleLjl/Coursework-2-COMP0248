# Product Context: Table Detection from 3D Point Clouds

## Problem Domain

This project addresses the challenge of automatically detecting and segmenting tables in indoor environments using 3D point cloud data. Table detection is a fundamental task in:

1. **Scene Understanding**: Identifying tables helps in understanding the layout and functional areas within indoor spaces.
2. **Robotic Navigation**: Tables are important landmarks and obstacles for indoor robots to recognize.
3. **Augmented Reality**: Table surfaces provide natural planes for placing virtual objects in AR applications.
4. **Smart Environments**: Detecting tables can help in contextualizing human activities in smart homes and offices.

## User Needs

This solution serves the needs of:

- **Computer Vision Researchers**: Who need benchmark implementations for 3D object detection
- **Robotics Engineers**: Who need reliable methods for identifying tables in indoor environments
- **AR/VR Developers**: Who need to identify horizontal surfaces for placing virtual content
- **Smart Environment Designers**: Who need to detect furniture and understand space usage

## Project Goals

The primary goals of this project are:

1. **Multiple Approach Comparison**: Implement and evaluate different pipelines for table detection to understand the strengths and limitations of each approach.
2. **Cross-Domain Learning**: Bridge 2D RGB images, depth maps, and 3D point clouds through different processing pipelines.
3. **Real-World Application**: Ensure models work not just on benchmark datasets but also on real-world data captured with commercial depth sensors (RealSense).
4. **Performance Evaluation**: Establish clear metrics and evaluation procedures to quantify the effectiveness of each pipeline.

## Success Criteria

The project will be considered successful if:

1. All three pipelines are implemented and functional
2. Models perform reasonably well on the test datasets (both Harvard sequences and RealSense captures)
3. Comprehensive evaluation metrics are provided for comparing pipeline performance
4. Strengths and weaknesses of each approach are clearly identified and documented
5. The code is modular, well-documented, and reusable
6. The report clearly explains the approaches, processing steps, and findings

## User Experience Goals

From a user perspective, the ideal solution should:

1. **Be Accurate**: Correctly identify tables in various indoor environments
2. **Be Generalizable**: Work across different settings, lighting conditions, and table types
3. **Be Efficient**: Process data within reasonable time constraints
4. **Be Interpretable**: Provide clear confidence scores for classification and visually interpretable segmentation results
5. **Be Robust**: Handle occlusions, varying distance from camera, and partial views of tables

## Constraints

The project operates within these constraints:

1. **Limited Dataset Size**:
   - Training: MIT sequences (290 frames)
   - Validation: Stratified random subset of Harvard sequences (48 frames)
   - Test Set 1: Remaining stratified random subset of Harvard sequences (50 frames)
   - Test Set 2: RealSense sequence (max 50 frames, to be collected)
   - This split allows for validation during training while maintaining an unseen test set (Test Set 1).
2. **Academic Context**: Focusing on methodology and evaluation rather than production-ready implementation
3. **Specific Table Definition**: Only certain furniture types (table top, dining table, desk, coffee table) are considered tables
4. **Fixed Evaluation Framework**: Must adhere to the coursework evaluation metrics and reporting format
5. **Generalization Challenge**: Models need to generalize across different environments (MIT vs Harvard sequences) which have different characteristics
