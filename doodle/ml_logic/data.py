import os
import numpy as np
from keras.applications.vgg16 import preprocess_input

from keras.utils import image_dataset_from_directory

def preprocess(ds):
    batch_images = next(iter(ds))[0]
    processed_images = preprocess_input(batch_images)
    labels = next(iter(ds))[1]
    return processed_images, labels


def get_data(data_size):
    """
    Gets a dataset object from the dataset directory to be passed on to the model
    """


    cwd = os.getcwd()

    if data_size == 'small':
        data_dir = os.path.join(cwd, 'small_data')
    elif data_size == 'aug':
        data_dir = os.path.join(cwd, 'data_aug')
    elif data_size == '20':
        data_dir = os.path.join(cwd, 'data_20')
    elif data_size == '20_a':
        data_dir = os.path.join(cwd, 'data_20_a')
    elif data_size == '50_a':
        data_dir = os.path.join(cwd, 'data_50_a')
    else:
        data_dir = os.path.join(cwd,'data')

    batch_size = 32
    image_size = (64,64)

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode='int',
        labels = 'inferred',
        subset="training",
        seed=123,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode='int',
        labels='inferred',
        subset="validation",
        seed=123,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds
