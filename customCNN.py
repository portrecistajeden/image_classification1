import keras
import tqdm as tqdm
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Used to create model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Used to show graphs
import numpy as np
import matplotlib.pyplot as plt

from data_prep import load_dataCustom


class customCNN:
    def __init__(self, trainDir, testDir):
        # Load data form data_prep file
        self.train_data = load_dataCustom(trainDir)
        self.test_data = load_dataCustom(testDir)

        # Useful variables
        # To do: let user change this
        self.classes = 6
        self.epochs = 2
        self.width = 100
        self.height = 100

        self.step_size_train = self.train_data.n // self.train_data.batch_size
        self.step_size_test = self.test_data.n // self.test_data.batch_size

        self.loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.SGD(learning_rate=0.01)

        self.createModel()
        self.customFit(self.epochs)

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


    def loss(self, model, x, y, training):
            y_ = model(x, training=training)
            return self.loss_object(y_true=y, y_pred=y_)


    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    def customFit(self, epochs):     
        train_loss_results = []
        train_accuracy_results = []

        for epoch in range(self.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            print("hewwo")
            for x, y in self.train_data:

                loss_value, grads = self.grad(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                epoch_loss_avg.update_state(loss_value)  
                epoch_accuracy.update_state(y, self.model(x, training=True))

            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 2 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))

        print(train_loss_results)
        print(train_accuracy_results)

                                                                        