import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def load_images(directory):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(100, 100),
        shuffle=True,
        interpolation="bilinear"
    )
    return data


def load_data(dirPath):
    data = ImageDataGenerator().flow_from_directory(
        directory = dirPath, #
        target_size = (100, 100),
        color_mode = 'rgb',
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True,
        seed = 10010
    )

    return data
