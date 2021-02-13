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
    def __init__(self, trainDir, testDir, epochs, console):
        # Load data form data_prep file
        self.train_data = load_dataCustom(trainDir)
        self.test_data = load_dataCustom(testDir)

        # Useful variables
        # To do: let user change this
        self.classes = getNumberOfClasses(trainDir)
        self.epochs = epochs
        self.width = 100
        self.height = 100

        # optimizer = keras.optimizers.Adam()
        # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # acc_metric = keras.metrics.SparseCategoricalAccuracy()

        self.createModel()
        # self.customTraining(epochs, optimizer, loss_fn, acc_metric)
        # self.customTest(acc_metric)
        training = CustomFit(self.model)
        training.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            acc_metric=keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        )
        training.fit(self.train_data, batch_size=32, epochs=self.epochs)
        training.evaluate(self.test_data, batch_size=32)


    def createModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu',))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.2))
        #
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))
        #
        # self.model.add(Conv2D(128, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        #
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))

        self.model.add(Dense(self.classes, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # def customTraining(self, epochs, optimizer, loss_fn, acc_metric, console):
    #
    #     for epoch in range(epochs):
    #         console.append(f"\n Epoch {epoch}")
    #         for batch_idx, (x, y) in enumerate(self.train_data):
    #             with tf.GradientTape() as tape:
    #                 pred = self.model(x, training=True)
    #                 loss = loss_fn(y, pred)
    #
    #             gradients = tape.gradient(loss, self.model.trainable_weights)
    #             optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    #             acc_metric.update_state(y, pred)
    #
    #         train_acc = acc_metric.result()
    #         console.append(f"Accuracy {train_acc}")
    #         acc_metric.reset_states()
    #
    # def customTest(self, acc_metric):
    #     for batch_idx, (x, y) in enumerate(self.test_data):
    #         pred = self.model(x, training=True)
    #         acc_metric.update_state(y, pred)
    #     train_acc = acc_metric.result()
    #     print(f"Test accuracy {train_acc}")
    #     acc_metric.reset_states()
    #
    # def accGraph(self, accPlot):
    #     arr = np.arange(0, self.epochs)
    #     accPlot.plot(arr, self.history.history["accuracy"], label="train_acc")
    #     accPlot.plot(arr, self.history.history["val_accuracy"], label="val_acc")
    #     #accPlot.title("Training accuracy")
    #     #accPlot.xlabel("epoch #")
    #     #accPlot.ylabel("accuracy")
    #     accPlot.legend()