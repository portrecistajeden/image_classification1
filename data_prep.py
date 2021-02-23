import cv2
import numpy as np
import tensorflow as tf
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import os
from PIL import Image

target_size = (200, 200)

def getTargetSize():
    return target_size

def load_data(dirPath):
    data = ImageDataGenerator().flow_from_directory(
        directory = dirPath, #
        target_size = target_size,
        color_mode = 'rgb',
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True,
        seed = 10010
    )

    return data

def get_sorted_labels(dirPath):
    labels = []
    for entry in os.scandir(dirPath):
        labels.append(entry.name)
    labels.sort()
    return labels

def load_prediction_data(dirPath):
    predictData = []
    predictDataBatches = []
    for entry in os.scandir(dirPath):
        path = dirPath + '/' + entry.name
        image = tf.keras.preprocessing.image.load_img(
            path,
            target_size=target_size)
        input_arr = keras.preprocessing.image.img_to_array(image)
        predictData.append(input_arr)
        input_arr = np.array([input_arr])
        predictDataBatches.append(input_arr)
    return predictDataBatches, predictData

def load_dataCustom(dirPath):
    data = ImageDataGenerator().flow_from_directory(
        directory = dirPath, #
        target_size = target_size,
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
            img = img1.resize(target_size)
            img = np.array(img, dtype=np.float32)
            if len(img.shape) != 3:
                break
            img = imageTo1DVector(img)
            img = img.flatten()
            data.append(img)
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
            img = img1.resize(target_size)
            img = np.array(img, dtype=np.float32)
            if len(img.shape) != 3:
                break
            img = imageTo1DVector(img)
            #img = np.vstack(img).astype(np.float32)
            img = img.flatten()
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