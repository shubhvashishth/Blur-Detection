# Loading Dependencies

import numpy as np
import pandas as pd
import os
import pickle
import cv2

from sklearn.model_selection import train_test_split

from keras.utils import load_img
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout

# Pre-Processing

X_train = []
y_train = []

input_size = (96, 96)

folderpath = '/Users/shubhvashishth/Downloads/TrainingSet/Undistorted'

for filename in os.listdir(folderpath):
    imagepath = folderpath + '/' + filename
    img = load_img(imagepath, target_size = input_size)
    X_train.append((1/255)*np.asarray(img))
    # compute the Laplacian of the image and then return the focus
    img2 = cv2.imread(imagepath)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    blur_map = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.var(blur_map)
    y_train.append(score)
print("Trainset: Undistorted loaded...")

folderpath = '/Users/shubhvashishth/Downloads/TrainingSet/Artificially-Blurred'

for filename in os.listdir(folderpath):
    imagepath = folderpath + '/' + filename
    img = load_img(imagepath, target_size = input_size)
    X_train.append((1/255)*np.asarray(img))
    # compute the Laplacian of the image and then return the focus
    img2 = cv2.imread(imagepath)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    blur_map = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.var(blur_map)
    y_train.append(score)
print("Trainset: Artificially Blurred loaded...")

folderpath = '/Users/shubhvashishth/Downloads/TrainingSet/Naturally-Blurred'

for filename in os.listdir(folderpath):
    imagepath = folderpath + '/' + filename
    img = load_img(imagepath, target_size = input_size)
    X_train.append((1/255)*np.asarray(img))
    # compute the Laplacian of the image and then return the focus
    img2 = cv2.imread(imagepath)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    blur_map = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.var(blur_map)
    y_train.append(score)
print("Trainset: Naturally Blurred loaded...")


# Converting the training data to a pickel file

with open('X_train.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)

with open('y_train.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)


# Defining the CNN architecture

model = Sequential()

# Layer 1
model.add(Convolution2D(32, (5, 5), input_shape=(input_size[0], input_size[1], 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(64, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Layer 3
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Layer 4

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    range_val = max_val - min_val
    normalized_lst = [(x - min_val) / range_val for x in lst]
    return normalized_lst

# normalising target values between 0 and 1 
normal_y = normalize_list(y_train)

train_x = np.asarray(X_train)
train_y = np.asarray(normal_y)

x_training, x_testing, y_training, y_testing = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model.fit(x_training, y_training, batch_size=32, epochs=50, verbose=1)
print("Model training complete...")

test_loss, test_mae = model.evaluate(x_testing, y_testing)

# Calculate the accuracy score
accuracy_score = 1.0 / (1.0 + test_mae)
print("Accuracy Score:", accuracy_score)

# saving the model
model.save('model.h5')