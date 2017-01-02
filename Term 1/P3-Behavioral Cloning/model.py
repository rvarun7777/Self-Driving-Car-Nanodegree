from keras.layers import BatchNormalization, Convolution2D, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Nadam
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

pixel_height = int(config['input_size']['height'])
pixel_width = int(config['input_size']['width'])
color_channels = 3
target_size = (pixel_height, pixel_width)
input_shape = (pixel_height, pixel_width, color_channels)

# comma.ai's steering model (https://github.com/commaai/research/blob/master/train_steering_model.py)
# but with batchnorm as its first layer
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
optimizer = Nadam()
model.compile(optimizer=optimizer, loss='mse')

from training_validation import build_dataframes

training_samples, validation_samples = build_dataframes()

from dataframe_iterator import DataframeIterator

batch_size = 256
brightness_factor_min = 0.25
training_generator = DataframeIterator(training_samples, 
                                       brightness_factor_min=brightness_factor_min,
                                       translation=True,
                                       horizontal_flip=True, 
                                       target_size=target_size, 
                                       batch_size=batch_size)
validation_generator = DataframeIterator(validation_samples, 
                                         target_size=target_size, 
                                         batch_size=batch_size, 
                                         shuffle=False)

max_epochs = 8

training_samples_count = len(training_samples)
validation_samples_count = len(validation_samples)

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='model.weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
learning_rate_plateau_reducer = ReduceLROnPlateau(verbose=1, patience=0, epsilon=1e-5)

model.fit_generator(training_generator, training_samples_count,
                    max_epochs, 
                    callbacks=[model_checkpoint, learning_rate_plateau_reducer, early_stopping],
                    validation_data=validation_generator, 
                    nb_val_samples=validation_samples_count)

# Save model

import json

model_json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(model_json_string, outfile)