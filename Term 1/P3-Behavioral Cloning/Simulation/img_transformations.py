from PIL import Image, ImageEnhance
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

steering_correction = float(config['steering_correction']['adjustment_factor'])

def cropout_sky_hood(img, hood_pixel_size=25):
    """
    Crop out some of the sky and car's hood.
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1rfh1giot
    """
    height = img.height # expecting 160
    width = img.width # expecting 320
    img = img.crop((0, int(height / 5.0), width, height - hood_pixel_size))
    return img

def translate(image, steering_angle, steering_correction=steering_correction):
    x_translation_range = 80
    x_translation = x_translation_range*np.random.uniform() - (x_translation_range / 2.0)
    steering_angle = steering_angle - (x_translation / x_translation_range * 2.0 * steering_correction)
    y_translation_range = 40
    y_translation = y_translation_range*np.random.uniform() - y_translation_range / 2.0
    transformation_matrix = (1, 0, x_translation, 0, 1, y_translation)
    transformed_img = image.transform(image.size, Image.AFFINE, transformation_matrix)
    
    return transformed_img, steering_angle

def brighten_or_darken(img, brightness_factor_min=0.25):
    enhancer = ImageEnhance.Brightness(img)
    random_factor = brightness_factor_min + np.random.uniform()
    enhanced_img = enhancer.enhance(random_factor)
    return enhanced_img
