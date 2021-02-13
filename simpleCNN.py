# Placeholder for validation data
# To do: Change this later
from keras.preprocessing.image import ImageDataGenerator

# Used to create model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Used to show graphs
import numpy as np
import matplotlib.pyplot as plt

# Used to load data
from data_prep import load_data, getNumberOfClasses


class CNN():
    def __init__(self, trainDir, testDir, epochs):

        # Load data form data_prep file
        self.train_data = load_data(trainDir)
        self.test_data = load_data(testDir)

        # Useful variables
        # To do: let user change this
        self.classes = getNumberOfClasses(trainDir)
        self.epochs = epochs
        self.width = 100
        self.height = 100

        self.step_size_train = self.train_data.n // self.train_data.batch_size
        self.step_size_test = self.test_data.n // self.test_data.batch_size

        self.createModel()
        self.fitModel()
        # self.accGraph(accPlot)
        # self.lossGraph()
        # self.predictGraph(testDir)

    def createModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.height, self.width, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

        self.model.summary()
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def fitModel(self):
        self.history = self.model.fit(x=self.train_data, epochs=self.epochs, validation_data=self.test_data)

    def accGraph(self, accPlot):
        arr = np.arange(0, self.epochs)
        accPlot.plot(arr, self.history.history["accuracy"], label="train_acc")
        accPlot.plot(arr, self.history.history["val_accuracy"], label="val_acc")
        accPlot.set_title("Training accuracy")
        accPlot.set_xlabel("epoch #")
        accPlot.set_ylabel("accuracy")
        accPlot.legend()

    def lossGraph(self, lossPlot):
        arr = np.arange(0, self.epochs)
        lossPlot.plot(arr, self.history.history["loss"], label="train_loss")
        lossPlot.plot(arr, self.history.history["val_loss"], label="val_loss")
        lossPlot.set_title("Training loss")
        lossPlot.set_xlabel("epoch #")
        lossPlot.set_ylabel("loss")
        lossPlot.legend()

    # def predictGraph(self, testDir, predictPlot):
    #     # To do: move this to data_prep file
    #     score_generator = ImageDataGenerator().flow_from_directory(
    #         directory=testDir,
    #         target_size=(self.height, self.width),
    #         color_mode="rgb",
    #         batch_size=1,
    #         class_mode="categorical",
    #         shuffle=False
    #     )
    #
    #     predicted_values = self.model.predict(score_generator, steps=score_generator.n // score_generator.batch_size,
    #                                           verbose=1)
    #
    #     predicted_labels = []
    #
    #     for p in predicted_values:
    #         predicted_labels.append(np.argmax(p))
    #
    #     from sklearn.metrics import classification_report, confusion_matrix
    #     import itertools
    #     # print(classification_report(score_generator.classes, predicted_labels, target_names=test_data.class_indices))
    #
    #     # To do: make it more readable
    #     def plot_confusion_matrix(cm, classes,
    #                               title='Confusion matrix',
    #                               cmap=plt.cm.Blues):
    #
    #         predictPlot.title(title)
    #         predictPlot.colorbar()
    #         tick_marks = np.arange(len(classes))
    #         predictPlot.xticks(tick_marks, classes, rotation=45)
    #         predictPlot.yticks(tick_marks, classes)
    #
    #         fmt = 'd'
    #         thresh = cm.max() / 2.
    #         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #             predictPlot.text(j, i, format(cm[i, j], fmt),
    #                      horizontalalignment="center",
    #                      color="white" if cm[i, j] > thresh else "black")
    #
    #         predictPlot.ylabel('True label')
    #         predictPlot.xlabel('Predicted label')
    #         predictPlot.tight_layout()
    #
    #     cm = confusion_matrix(score_generator.classes, predicted_labels)
    #     plot_confusion_matrix(cm, self.test_data.class_indices)
    #     plt.show()
