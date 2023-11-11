#Loading Dependencies
import numpy as np
import pandas as pd
import os
import pickle
import cv2

import keras
from keras.utils import load_img

# loading the saved model
model = keras.models.load_model('model.h5')

# Pre-Processing the image for feeding to the model
input_size = (96, 96)
imagepath = '/Users/shubhvashishth/Downloads/TrainingSet/Undistorted/P1012598.JPG'
img = load_img(imagepath, target_size = input_size)
test_img = (1/255)*np.asarray(img)
test_img = np.expand_dims(test_img, axis=0)

# Predicting
x = model.predict(test_img)
val = x[0][0]

with open('y_train.pkl', 'rb') as picklefile:
    y_train = pickle.load( picklefile)


# De-normalising the predicted value
def denormalize(val, orig_lst):

    min_val = min(orig_lst)
    max_val = max(orig_lst)
    range_val = max_val - min_val
    denormalized_val = (x * range_val) + min_val
    return denormalized_val

# Function call
final_val = denormalize(val,y_train)

print("Higher the score is the better quality of image is")
print("Bluriness Score : ",final_val[0][0])