from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          MaxPooling2D)
from keras.models import Sequential

model = Sequential()

model.add(Flatten(input_shape=(224, 224))), # dimensions of the image
model.add(Conv2D(filters=64, kernel_size=(7, 7), stride=2))
model.add(MaxPooling2D(pool_size=(3, 3), stride=2))
