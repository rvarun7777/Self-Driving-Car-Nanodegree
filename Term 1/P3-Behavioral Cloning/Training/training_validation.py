import pandas as pd
from sklearn.model_selection import train_test_split
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

steering_correction = float(config['steering_correction']['adjustment_factor'])

# The driving log files doesn't include headers, so let's define them before we read the CSV file into a Pandas dataframe
column_names = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'brake', 'speed']

def steering_corrected_dataframe(dataframe, column, steering_correction):
    """
    Returns a new dataframe with the specified column (e.g. 'left_image' or 'right_image') of the original dataframe re-named to 
    'center_image'and the steering angle adjusted with the given steering correction.
    """
    dataframe_copy = dataframe[[column, 'steering_angle']].copy()
    dataframe_copy['steering_angle'] = dataframe_copy['steering_angle'].apply(lambda angle: float(angle) + steering_correction)
    dataframe_copy.columns = ['center_image', 'steering_angle']
    return dataframe_copy

parent_directories = [
    'data/track1/forward/center', # 'forward' & 'backward' == counter-clockwise & clockwise around the track
    'data/track1/backward/center'
]

def build_dataframes(parent_directories=parent_directories, steering_correction=steering_correction):
    """
    Build training dataframe by concatenating dataframes for different driving sessions, then split that into training & 
    validation dataframes
    """
    training_dataframe = None

    for parent_directory in parent_directories:
        dataframe = pd.read_csv(parent_directory + '/driving_log.csv', header=None, names=column_names, low_memory=False)
        left_image_steering_corrected = steering_corrected_dataframe(dataframe, 'left_image', steering_correction)
        right_image_steering_corrected = steering_corrected_dataframe(dataframe, 'right_image', -steering_correction)
        if type(training_dataframe) is pd.DataFrame:
            training_dataframe = pd.concat([training_dataframe, dataframe, left_image_steering_corrected, right_image_steering_corrected])
        else:
            training_dataframe = pd.concat([dataframe, left_image_steering_corrected, right_image_steering_corrected])

    shuffled_training_dataframe = training_dataframe.sample(frac=1) # shuffle training data

    training_samples, validation_samples = train_test_split(shuffled_training_dataframe)
    
    training_samples = training_samples.reset_index(drop=True)
    validation_samples = validation_samples.reset_index(drop=True)
    return training_samples, validation_samples
