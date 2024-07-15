# Pixelated Image Detection and Correction

This repository contains two main projects: 

1. **Image Classification**: Detect if an image is pixelated.
2. **Image Generation**: Improve the quality of a pixelated image.

## Problem Statement

### Part 1: Pixelated Image Detection
Given an image, check if it is blocky or pixelated. Pixelation refers to the blurry, blocky squares you see when an image is zoomed too much. The goal is to design an algorithm that can detect whether the image is pixelated or not. The algorithm or ML/AI model should be extremely lightweight and computationally efficient to run at 60 FPS with at least 90% accuracy. The pixelated images are rare, so the algorithm should minimize false positives. The model's quality will be measured by F1-score, precision-recall curve, etc., and it should handle 1080p resolution input.

- **Input Image Size**: 1920x1080 (downscaling is allowed)
- **Inference Speed Target**: Minimum 30 Hz, ideally 60 Hz
- **Accuracy Metric**: F1 Score / Precision-Recall
- **False Positives**: At most 10% false positives in rare class scenarios

### Part 2: Pixelated Image Restoration
Given a pixelated image, design an algorithm to improve its quality by restoring the lost information (JPEG restoration). The challenge is to design a highly efficient algorithm that can run at least at 20 FPS. The quality of restoration can be measured using metrics like LPIPS, PSNR, etc. If a non-pixelated image is given, the algorithm should leave it intact.

- **Target FPS**: At least 20 FPS
- **Image Quality Metrics**: LPIPS, PSNR, etc.
- **Resolution**: 1080p


## Detection

### detection.py

This script contains the implementation for detecting pixelated images. The process includes:

1. **Data Splitting**: Split data into training and validation sets.
2. **Image Augmentation**: Use `ImageDataGenerator` for data augmentation.
3. **Model Definition**: Define a CNN model for binary classification.
4. **Training**: Train the model with callbacks for early stopping, learning rate reduction, and model checkpointing.
5. **Evaluation**: Evaluate the model's performance on the test set.

#### Key Functions and Classes

- `split_data(original_dir, pixelated_dir, train_dir, test_dir, split_ratio)`: Splits the data into training and testing sets.
- `train_datagen` and `validation_datagen`: Image data generators for augmentation and scaling.
- `model`: A Sequential CNN model for classification.
- `early_stopping`, `reduce_lr`, `model_checkpoint`: Callbacks for model training.
- `model.fit()`: Training the model.
- `model.evaluate()`: Evaluating the model.

### Usage

1. Define the paths to your dataset folders.
2. Run `detection.py` to train and evaluate the model.
3. Use the trained model to predict whether an image is pixelated or not.

## Correction

### correction.py

This script contains the implementation for restoring pixelated images. The process includes:

1. **Noise Addition**: Add Gaussian or salt-and-pepper noise to images.
2. **Denoising Methods**: Apply various denoising methods (Gaussian, median, bilateral, Non-local Means).
3. **PSNR Calculation**: Calculate PSNR for each denoising method.
4. **Comparison**: Compare the performance of different denoising methods.

#### Key Functions and Classes

- `add_noise(image, noise_type)`: Adds noise to the image.
- `gaussian_filter(image, kernel_size)`, `median_filter(image, kernel_size)`, `bilateral_filter(image, d, sigma_color, sigma_space)`, `nlm_filter(image)`: Various denoising methods.
- `psnr(img1, img2)`: Calculates PSNR between two images.
- `compare_denoising_methods()`: Compares different denoising methods and displays results.

### Usage

1. Upload an image.
2. Run `correction.py` to compare different denoising methods and visualize the results.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- scikit-image
- matplotlib

## Team Members

|  Name       | Role                                                         |
|-------------------|--------------------------------------------------------------|
| Jagriti Sharma    | Project Lead, Correction Model Development                   |
| Harshita Agrawal  | Testing and Integration, Correction Model Development        |
| Aayush Dharpure   | Testing and Integration, Presentation, Detection Model Development |
| Prathamesh Rokade | Testing, Researcher, Documentation, Detection Model Development |



## Installation

To install the required packages, run:

```bash
pip install tensorflow numpy opencv-python scikit-image matplotlib
