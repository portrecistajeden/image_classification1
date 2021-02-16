import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import os
from PIL import Image

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
        target_size = (400, 400),
        color_mode = 'rgb',
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True,
        seed = 10010
    )

    return data

def load_dataCustom(dirPath):
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
    data = []
    labelsNum = []
    dictionary = {}
    i = 0

    for entry in os.scandir(dirPath):
        path = dirPath + '/' + entry.name
        for entry2 in os.scandir(path):
            imgPath = path + '/' + entry2.name
            img1 = Image.open(imgPath)
            img = img1.resize((400, 400))
            img = np.array(img, dtype=np.int)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if len(img.shape) != 3:
                break
            img = imageTo1DVector(img)
            #imgStacked = np.vstack(img).astype(np.float32)
            #imgStacked = imgStacked.flatten()
            #img = img[:, :, 0]
            img = img.flatten()
            #print(img)
            data.append(img)
            if img.shape != (480000,):
                print(imgPath)
                print(f"img shape: {img.shape}")
            labelsNum.append(i)

        dictionary[entry.name] = i
        i = i + 1
    data = np.array(data, dtype=np.float32)
    labelsNum = np.array(labelsNum, dtype=np.float32)
    return data, labelsNum, dictionary

def load_test_KNN(dirPath):
    data = []
    labelsStr = []
    for entry in os.scandir(dirPath):
        path = dirPath + '/' + entry.name
        for entry2 in os.scandir(path):
            imgPath = path + '/' + entry2.name
            img1 = Image.open(imgPath)
            img = img1.resize((400, 400))
            img = np.array(img, dtype=np.float32)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if len(img.shape) != 3:
                break
            img = imageTo1DVector(img)
            #img = np.vstack(img).astype(np.float32)
            img = img.flatten()
            if img.shape != (480000,):
                print(imgPath)
                print(f"img shape: {img.shape}")
            data.append(img)
            labelsStr.append(entry.name)

    data = np.array(data, dtype=np.float32)
    labelsStr = np.array(labelsStr, dtype=np.str)

    return data, labelsStr

def getNumberOfClasses(dirPath):
    i = 0
    for entry in os.scandir(dirPath):
        i += 1
    return i

def getTestLabels(dirPath):
    classes = []
    items = []
    for entry in os.scandir(dirPath):
        classes.append(entry.name)
        path = dirPath + '/' + entry.name
        for entry2 in os.scandir(path):
            items.append(entry.name)
    classes = np.array(classes, dtype=np.str)
    return classes, items

def getTrainingLabels(dirPath):
    classes = []
    for entry in os.scandir(dirPath):
        classes.append(entry.name)
    classes = np.array(classes, dtype=np.str)
    return classes

def imageTo1DVector(img):
    length, height, depth = img.shape
    return img.reshape((length * height * depth, 1))