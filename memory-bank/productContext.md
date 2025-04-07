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

*(Scope revised to focus solely on Pipeline A)*

The primary goals of this project are:

1. **Implement Pipeline A**: Successfully implement Pipeline A (Depth -> Point Cloud -> Classification) for table detection.
2. **Real-World Application**: Ensure Pipeline A works not just on benchmark datasets but also on real-world data captured with commercial depth sensors (RealSense/UCL dataset).
3. **Performance Evaluation**: Establish clear metrics and evaluation procedures to quantify the effectiveness of Pipeline A.

## Success Criteria

*(Scope revised to focus solely on Pipeline A)*

The project will be considered successful if:

1. Pipeline A is implemented and functional.
2. The Pipeline A model performs reasonably well on the test datasets (both Harvard sequences and the 'ucl' dataset).
3. Comprehensive evaluation metrics are provided for Pipeline A's performance.
4. Strengths and weaknesses of Pipeline A are clearly identified and documented.
5. The code for Pipeline A is modular, well-documented, and reusable.
6. The report clearly explains the Pipeline A approach, processing steps, and findings.

## User Experience Goals

From a user perspective, the ideal solution should:

1. **Be Accurate**: Correctly identify tables in various indoor environments
2. **Be Generalizable**: Work across different settings, lighting conditions, and table types
3. **Be Efficient**: Process data within reasonable time constraints (for Pipeline A).
4. **Be Interpretable**: Provide clear confidence scores for classification (from Pipeline A).
5. **Be Robust**: Handle occlusions, varying distance from camera, and partial views of tables (within Pipeline A's capabilities).

## Constraints

The project operates within these constraints:

1. **Limited Dataset Size**:
   - Training: MIT sequences (290 frames)
   - Validation: Stratified random subset of Harvard sequences (48 frames)
   - Test Set 1: Remaining stratified random subset of Harvard sequences (50 frames)
   - Test Set 2: Custom 'ucl' dataset (RealSense capture, size varies)
   - This split allows for validation during training while maintaining unseen test sets (Test Set 1 and Test Set 2) for final evaluation.
2. **Academic Context**: Focusing on methodology and evaluation rather than production-ready implementation
3. **Specific Table Definition**: Only certain furniture types (table top, dining table, desk, coffee table) are considered tables
4. **Fixed Evaluation Framework**: Must adhere to the coursework evaluation metrics and reporting format
5. **Generalization Challenge**: Models need to generalize across different environments (MIT vs Harvard sequences) which have different characteristics
