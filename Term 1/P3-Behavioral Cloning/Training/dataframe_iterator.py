"""
Inheriting from Keras' Iterator class, this class generates batches of images from the Pandas dataframe given in the constructor.
"""

from keras.preprocessing.image import Iterator, load_img, img_to_array, flip_axis
import numpy as np
from  pandas import DataFrame
from img_transformations import cropout_sky_hood, translate, brighten_or_darken
from PIL import Image

class DataframeIterator(Iterator):

    def __init__(self, dataframe, brightness_factor_min=None, translation=False, horizontal_flip=False,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='tf',
                 batch_size=32, shuffle=True, seed=None):

        self.dataframe = dataframe
        self.brightness_factor_min=brightness_factor_min
        self.translation = translation
        self.horizontal_flip = horizontal_flip
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2
            self.img_row_index = self.row_index - 1
            self.img_col_index = self.col_index - 1
            self.img_channel_index = self.channel_index - 1
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.nb_sample = len(dataframe)
        super(DataframeIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = np.zeros(current_batch_size)
        # build batch of image data
        for i, j in enumerate(index_array):
            current_dataframe = self.dataframe.iloc[j]
            image_filepath = current_dataframe['center_image'].strip() # NOTE: parameterize column name
            img = load_img(image_filepath)#, target_size=self.target_size)
            
            if self.brightness_factor_min:
                img = brighten_or_darken(img, self.brightness_factor_min)
            
            y = current_dataframe['steering_angle']
            if self.translation:
                img, y = translate(img, y)
            
            img = cropout_sky_hood(img)
            
            img = img.resize((self.target_size[1], self.target_size[0]))
            
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            
            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    x = flip_axis(x, self.img_col_index)
                    y = -y # flip steering angle when we horizontally flip image
            batch_x[i] = x
            batch_y[i] = y
        return batch_x, batch_y
