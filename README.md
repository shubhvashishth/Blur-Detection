
# Image Blur Detection using Convolutional Neural Networks

This project employs Convolutional Neural Networks (CNNs) to detect and quantify image blur. By training on three distinct datasets—Undistorted, Artificially Blurred, and Naturally Blurred images—the model predicts a blurriness score based on the Laplacian map variance. The aim is to provide a robust solution for image quality assessment.


## Table of Contents


- Overview
- Run Locally
- Training
- Testing
- Usage
- Project Structure
- Technical Details
- Data Pre-processing
- CNN Architecture
- Normalization and Denormalization


## Run Locally

- Clone the project

- To replicate the project environment, utilize the provided requirements.txt file. It outlines the essential dependencies for running the scripts.

```bash
  git clone https://github.com/shubhvashishth/Blur-Detection
```




## Training

The `train.py` script loads images from three categories, computes the Laplacian map variance, and trains the CNN model. The trained model is serialized as `model.h5`, while the training data is stored in `X_train.pkl` and `y_train.pkl`.
## Testing

Use the `test_model.py` script to load the trained model and assess a new image's blurriness score. The script provides valuable insights into the image quality.


## Usage

1. **Training:**
Execute the train.py script to train the CNN model.

2. **Testing:**
Update the imagepath variable in test_model.py with the path to the image for testing.
Run test_model.py to obtain the blurriness score for the tested image.


## Project Structure 

`train.py`: Script for training the CNN model using the provided datasets.

`test_model.py`: Script for testing the trained model on a new image.

`model.h5`: Serialized trained model.

`X_train.pkl`: Serialized training data (input images).

`y_train.pkl`: Serialized training labels (blurriness scores).
requirements.txt: File specifying project dependencies.


## Technical Details

**Data Pre-processing :**
The training data undergoes pre-processing, including loading images, resizing, and computing the Laplacian of each image to extract focus information.

**CNN Architecture :**
The CNN architecture consists of multiple layers, including convolutional, activation, pooling, flatten, and dense layers. The model is designed to capture features relevant to image blur.

**Normalization and Denormalization :**
Training data and predicted values are normalized to facilitate model training. The normalize_list function scales values between 0 and 1. During testing, the denormalize function is employed to obtain the original blurriness score from the normalized prediction.