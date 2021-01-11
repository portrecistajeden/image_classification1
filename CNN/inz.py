from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf

# from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback

#potrzebne zmienne:
#liczba klas
classes = 6 #np jablka, pomidory, buraki i gruszki
#szerokosc/wysokosc obrazow
width, height = 100,100
#ilosc epok naucznia
epochs = 30


#wczytanie obrazow oraz podzial na grupy terningowe i testowe

train_data = ImageDataGenerator().flow_from_directory(
    directory = 'fruits-360/Training', #
    target_size = (height, width),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    seed = 10010
)

test_data = ImageDataGenerator().flow_from_directory(
    directory = 'fruits-360/Test', #
    target_size = (height, width),
    color_mode = 'rgb',
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    seed = 10010
)


step_size_train=train_data.n//train_data.batch_size
step_size_test=test_data.n//test_data.batch_size


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

model.summary()


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# with tf.device('GPU:1'):

history = model.fit(x=train_data,
                    epochs=epochs,
                    validation_data=test_data)


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# matplotlib.use('Agg')

# %matplotlib inline
arr = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(arr, history.history["accuracy"], label="train_acc")
plt.plot(arr, history.history["val_accuracy"], label="val_acc")
plt.title("Training accuracy")
plt.xlabel("epoch #")
plt.ylabel("accuracy")
plt.legend()
plt.show()


arr = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(arr, history.history["loss"], label="train_loss")
plt.plot(arr, history.history["val_loss"], label="val_loss")
plt.title("Training loss")
plt.xlabel("epoch #")
plt.ylabel("loss")
plt.legend()
plt.show()


score_generator = ImageDataGenerator().flow_from_directory(
    directory='fruits-360/Validation',
    target_size=(height, width),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)


predicted_values = model.predict_generator(generator=score_generator, steps=score_generator.n//score_generator.batch_size, verbose=1)


predicted_labels = []

for p in predicted_values:
    predicted_labels.append(np.argmax(p))


from sklearn.metrics import classification_report, confusion_matrix
import itertools
print(classification_report(score_generator.classes, predicted_labels, target_names=test_data.class_indices))


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cm = confusion_matrix(score_generator.classes, predicted_labels)


plot_confusion_matrix(cm, test_data.class_indices)

plt.show()