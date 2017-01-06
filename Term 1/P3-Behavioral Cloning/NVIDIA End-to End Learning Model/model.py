import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import csv
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization


num_images = 0
image_files = []
steer_angles = []
use_left_right_cameras = False
print("Finding images and steering angles")

with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        
        center_image, left_image, right_image, center_angle = row[:4]
        center_angle = np.float32(center_angle)

        image_files.append(center_image.strip())
        steer_angles.append(center_angle)
        num_images += 1
            
        if use_left_right_cameras:
            image_files.append(right_image.strip())
            steer_angles.append(center_angle-0.2) # +0.2 crashes to left on bridge, 0 crashes sharp after bridge

            image_files.append(left_image.strip())
            steer_angles.append(center_angle+0.2) # -0.2 crashes to left on bridge, 0 crashes sharp after bridge
            num_images += 2



steer_angles=np.float32(steer_angles)


# assemble training data 
print("Collecting and formatting training data")
rows=160
cols=320
channels=3
X_train_raw = np.zeros((num_images,rows,cols,channels))
for i,file in enumerate(image_files):
    img = plt.imread(file)
    X_train_raw[i,:,:,:] = img


# assemble outputs
y_train_raw = steer_angles
y_train_raw = y_train_raw.reshape(-1,1) # need 2d shape, not just 1d array



# Normalize images, zero mean range -1 to +1
print("Normalizing input data")

X_train_raw = (X_train_raw-128.)/128.

# randomize order of images and create training and validation
print("Splitting data into training and validation sets")
X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw, test_size=0.3)



# Create Keras model, based on Nvidia Convolutional Neural Network
print("Creating Convolutional Neural Network model")

model = Sequential()

#scale image from 160x320 to 80x160
model.add(Convolution2D(nb_row=1, nb_col=1, border_mode='valid', 
                        nb_filter=3, init='normal',
                       subsample=(5,5), input_shape=(rows, cols, channels),
                        name='subsample'))
# 5x5 with 2x2 striding          
model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='valid', 
                        nb_filter=24,
                           activation='relu',
                          init='normal'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='valid', 
                        nb_filter=36,
                           activation='relu',
                          init='normal'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

# Nvidia model includes 3rd conv layer, which we don't use
# model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='same', 
#                         nb_filter=48,
#                            activation='relu', subsample=(4,4), 
#                           init='normal'))
# model.add(BatchNormalization())

# 3x3 with no striding
model.add(Convolution2D(nb_row=3, nb_col=3, border_mode='valid', 
                        nb_filter=64,
                           activation='relu',  
                          init='normal'))
model.add(BatchNormalization())

model.add(Convolution2D(nb_row=3, nb_col=3, border_mode='valid', 
                        nb_filter=64,
                           activation='relu',  
                          init='normal'))
model.add(BatchNormalization())

model.add(Flatten(name='flatten'))

model.add(Dense(output_dim=1164, init='normal', activation='relu'))
model.add(Dropout(p=0.5))
model.add(BatchNormalization())

model.add(Dense(output_dim=100, init='normal', activation='relu'))
model.add(Dropout(p=0.5))
model.add(BatchNormalization())

model.add(Dense(output_dim=50, init='normal', activation='relu'))
model.add(Dropout(p=0.5))
model.add(BatchNormalization())

model.add(Dense(output_dim=10, init='normal', activation='relu'))
model.add(Dropout(p=0.5))
model.add(BatchNormalization())

model.add(Dense(output_dim=1, name='output', init='normal'))

print(model.summary())


# compile and fit model
print("Fitting model")
model.compile(loss='mse', metrics=['mse'], optimizer=Adam())

fit = model.fit(X_train, y_train, batch_size=40,nb_epoch=10,verbose=1,validation_data=(X_valid, y_valid) )

# compare model predicted steering angles with labeled values
y_train_predict = model.predict(X_train)
print(y_train_predict.shape)

np.set_printoptions(suppress=True)
print(y_train_predict[0:40].T)
print(y_train[0:40].T)


# output model
print("Saving model structure and weights")
model_json = model.to_json()
import json
with open ('model.json', 'w') as f:
    json.dump(model_json, f, indent=4, sort_keys=True, separators=(',', ':'))

model.save_weights('model.h5')