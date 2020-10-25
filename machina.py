#importy kerasa i tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers

#potrzebne zmienne:
#liczba klas
classes = 3 #np jablka, pomidory, buraki i gruszki
#szerokosc/wysokosc obrazow
width, height = 128,128
#ilosc epok naucznia
epochs = 20


#wczytanie obrazow oraz podzial na grupy terningowe i testowe

# train_data = ImageDataGenerator().flow_from_directory(
#     directory = None, #
#     target_size = (height, width),
#     color_mode = 'rgb',
#     batch_size = 32,
#     class_mode = 'categorical',
#     shuffle = True,
#     seed = 10010
# )

# test_data = ImageDataGenerator().flow_from_directory(
#     directory = None, #
#     target_size = (height, width),
#     color_mode = 'rgb',
#     batch_size = 32,
#     class_mode = 'categorical',
#     shuffle = True,
#     seed = 10010
# )

#utworzenie modelu sekwencyjnego
model = keras.Sequential(
    [
        #utworzenie warst konwolucyjnych 
        layers.Conv2D(32, (3,3), activation='relu'), #where 32 is the number of filters the layer will learn
        layers.add(layers.MaxPooling2D((2, 2))),
        layers.Conv2D(64, (3,3), activation='relu'), #where (3,3) is the kernel size
        layers.add(layers.MaxPooling2D((2, 2))),
        layers.Conv2D(64, (3,3), activation='relu'), #where 'relu' is the linear activation function max(x, 0)
        #oraz kolejne warstwy w razie potrzeby
        layers.add(layers.Flatten()),
        layers.add(layers.Dense(64, activation='relu')),
        layers.add(layers.Dense(classes))
    ]
)

#mozliwosc uzycia model.summary() by wyswietlic wszystkie warstwy wraz z opisem
input_shape = (None, 32, 32, 3) #shape of input data
model.build(input_shape) #building the model
model.summary()

#kompilacja modelu
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy '])

#model.fit_generator(...) to start the learning process

print('pass')