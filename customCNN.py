import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Used to create model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Used to show graphs
import numpy as np
import matplotlib.pyplot as plt

from data_prep import load_dataCustom, getNumberOfClasses


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

        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
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
    def __init__(self, trainDir, testDir, console):
        # Load data form data_prep file
        self.train_data = load_dataCustom(trainDir)
        self.test_data = load_dataCustom(testDir)

        # Useful variables
        # To do: let user change this
        self.classes = getNumberOfClasses(trainDir)
        self.width = 400
        self.height = 400

    def trainModel(self, epochs):
        self.epochs = epochs
        self.training = CustomFit(self.model)
        self.training.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            acc_metric=keras.metrics.CategoricalAccuracy(name='accuracy'),
        )
        self.history = self.training.fit(self.train_data, batch_size=32, epochs=self.epochs)
        # print(self.history)

    def evaluateModel(self):
        self.training.evaluate(self.test_data, batch_size=32)

    def loadModel(self, path):
        self.model = tf.keras.models.load_model(path)

    def saveModel(self, path):
        self.model.save(path)

    def createModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='relu',))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        #
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))
        #
        # self.model.add(Conv2D(128, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        # self.model.add(Dropout(0.5))
        #
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.classes, activation='softmax'))


        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def accGraph(self, accPlot):
        arr = np.arange(1, self.epochs+1)
        accPlot.plot(arr, self.history.history["accuracy"], label="train_acc")
        # accPlot.plot(arr, self.history.history["val_accuracy"], label="val_acc")
        accPlot.set_title("Training accuracy")
        accPlot.set_xlabel("epoch #")
        accPlot.set_ylabel("accuracy")
        accPlot.set_xlim([1, self.epochs+1])
        accPlot.legend()

    def lossGraph(self, lossPlot):
        arr = np.arange(1, self.epochs+1)
        lossPlot.plot(arr, self.history.history["loss"], label="train_loss")
        # lossPlot.plot(arr, self.history.history["val_loss"], label="val_loss")
        lossPlot.set_title("Training loss")
        lossPlot.set_xlabel("epoch #")
        lossPlot.set_ylabel("loss")
        lossPlot.set_xlim([1, self.epochs+1])
        lossPlot.legend()


