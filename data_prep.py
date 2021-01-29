import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

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

def load_data_KNN(dirPath):
    cells = []
    labels = []
    i = 0

    for entry in os.scandir(dirPath):
        path = dirPath + '/' + entry.name
        for entry2 in os.scandir(path):
            imgPath = path + '/' + entry2.name
            img = cv2.imread(imgPath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cell = gray.flatten()
            cells.append(cell)
            #labels.append(entry.name)
            labels.append(i)
        i = i + 1

    cells = np.array(cells, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return cells, labels

def getNumberOfClasses(dirPath):
    dirs, files = os.walk(dirPath)
    return len(dirs[1])