import tqdm as tqdm
from keras.preprocessing.image import ImageDataGenerator

# Used to create model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Used to show graphs
import numpy as np
import matplotlib.pyplot as plt

from data_prep import load_data


class customCNN:
    def __init__(self, trainDir, testDir):
        # Load data form data_prep file
        self.train_data = load_data(trainDir)
        self.test_data = load_data(testDir)

        # Useful variables
        # To do: let user change this
        self.classes = 6
        self.epochs = 10
        self.width = 100
        self.height = 100

        self.step_size_train = self.train_data.n // self.train_data.batch_size
        self.step_size_test = self.test_data.n // self.test_data.batch_size

        self.createModel()
        self.customFitModel()

    def createModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.classes, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def customFit(self, epochs):
        for epoch in range(epochs):
            print('Epoch %d' % epoch)
            pbar = tqdm(range(len(self.train_data))) #to jest fancy progress bar, jeszcze ogarnę jak go wyswietlać