import io
import random
from contextlib import redirect_stdout

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Used to create model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Used to show graphs
import numpy as np
import matplotlib.pyplot as plt

from data_prep import *


class CustomFit(keras.Model):
    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def compile(self, optimizer, loss, acc_metric):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metric = acc_metric

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss_value = self.loss(y, pred)

        #compute gradients
        trainable_weights = self.model.trainable_weights
        gradients = tape.gradient(loss_value, trainable_weights)
        #update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        self.acc_metric.update_state(y, pred)

        return {"loss": loss_value, "accuracy": self.acc_metric.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        pred = self.model(x, training=True)
        loss_value = self.loss(y, pred)
        self.acc_metric.update_state(y, pred)

        return {"loss": loss_value, "accuracy": self.acc_metric.result()}

class CustomCNN:
    def __init__(self, trainDir, testDir, predictDir, optimizer, console):
        # Load data form data_prep file
        self.train_data = load_dataCustom(trainDir)
        self.validation_data = load_dataCustom(testDir)
        self.prediction_data_batches, self.prediction_data = load_prediction_data(predictDir)
        self.sorted_labels = get_sorted_labels(trainDir)
        self.console = console
        self.optimizer = optimizer
        # Useful variables
        # To do: let user change this
        self.classes = getNumberOfClasses(trainDir)
        self.width = 100
        self.height = 100

    def trainModel(self, epochs):
        self.epochs = epochs
        # self.training = CustomFit(self.model)
        # self.training.compile(
        #     optimizer=keras.optimizers.Adam(),
        #     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        #     acc_metric=keras.metrics.CategoricalAccuracy(name='accuracy'),
        # )
        # self.history = self.training.fit(
        #     self.train_data,
        #     batch_size=32,
        #     epochs=self.epochs,
        #     validation_data=self.validation_data)
        self.history = self.model.fit(x=self.train_data, epochs=self.epochs, validation_data=self.validation_data)

    def evaluateModel(self):
        f = io.StringIO()
        with redirect_stdout(f):
            self.training.evaluate(self.validation_data, batch_size=32)
        self.console.append(f.getvalue())

    def loadModel(self, path):
        self.model = tf.keras.models.load_model(path)

    def saveModel(self, path):
        self.model.save(path)

    def createModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(Conv2D(64, (3, 3), activation='relu',))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu', ))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu', ))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))

        self.model.add(Dense(self.classes, activation='softmax'))

        self.model.summary(print_fn=lambda x: self.console.append(x))
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    def accGraph(self, accPlot):
        arr = np.arange(1, self.epochs+1)
        accPlot.plot(arr, self.history.history["accuracy"], label="train_acc")
        accPlot.plot(arr, self.history.history["val_accuracy"], label="val_acc")
        accPlot.set_title("Training accuracy")
        accPlot.set_xlabel("epoch #")
        accPlot.set_ylabel("accuracy")
        accPlot.set_xlim([1, self.epochs])
        accPlot.legend()

    def lossGraph(self, lossPlot):
        arr = np.arange(1, self.epochs+1)
        lossPlot.plot(arr, self.history.history["loss"], label="train_loss")
        lossPlot.plot(arr, self.history.history["val_loss"], label="val_loss")
        lossPlot.set_title("Training loss")
        lossPlot.set_xlabel("epoch #")
        lossPlot.set_ylabel("loss")
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

