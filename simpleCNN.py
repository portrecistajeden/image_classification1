import io
import random
from contextlib import redirect_stdout

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from data_prep import *

class CNN:
    def __init__(self, trainDir, validationDir, predictDir, optimizer, console):

        # Load data form data_prep file
        self.train_data = load_data(trainDir)
        self.validation_data = load_data(validationDir)
        self.prediction_data_batches, self.prediction_data = load_prediction_data(predictDir)
        self.sorted_labels = get_sorted_labels(trainDir)
        self.console = console
        self.optimizer = optimizer

        self.classes = getNumberOfClasses(trainDir)
        self.width = 200
        self.height = 200

        self.step_size_train = self.train_data.n // self.train_data.batch_size
        self.step_size_test = self.validation_data.n // self.validation_data.batch_size


    def createModel(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

        self.model.summary(print_fn=lambda x: self.console.append(x))
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    def trainModel(self, epochs):
        self.epochs = epochs

        self.history = self.model.fit(x=self.train_data, epochs=self.epochs, validation_data=self.validation_data)

    def evaluateModel(self):
        f = io.StringIO()
        with redirect_stdout(f):
            self.results = self.model.evaluate(self.validation_data, batch_size=32)
        self.console.append(f.getvalue())

    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = tf.keras.models.load_model(path)

    def accGraph(self, accPlot):
        arr = np.arange(1, self.epochs+1)
        accPlot.plot(arr, self.history.history["accuracy"], label="train_acc")
        accPlot.plot(arr, self.history.history["val_accuracy"], label="val_acc")
        accPlot.set_title("Training accuracy")
        accPlot.set_xlabel("Epoch")
        accPlot.set_ylabel("Accuracy")
        accPlot.set_xlim([1, self.epochs])
        accPlot.legend()

    def lossGraph(self, lossPlot):
        arr = np.arange(1, self.epochs+1)
        lossPlot.plot(arr, self.history.history["loss"], label="train_loss")
        lossPlot.plot(arr, self.history.history["val_loss"], label="val_loss")
        lossPlot.set_title("Training loss")
        lossPlot.set_xlabel("Epoch")
        lossPlot.set_ylabel("Loss")
        lossPlot.set_xlim([1, self.epochs])
        lossPlot.legend()

    def predictGraph(self, predictGraph):
        labels_str = []
        for image in self.prediction_data_batches:
            y_prob = self.model.predict(image)
            y_classes = y_prob.argmax(axis=-1)
            labels_str.append(self.sorted_labels[y_classes[0]])

        r = random.sample(range(0, len(self.prediction_data)), k=6)
        ind = 0
        for x in range(2):
            for y in range(3):
                i = r[ind]
                ind += 1
                predictGraph[x, y].set_title(labels_str[i])
                predictGraph[x, y].imshow(tf.keras.preprocessing.image.array_to_img(self.prediction_data[i]))
                predictGraph[x, y].axis('off')