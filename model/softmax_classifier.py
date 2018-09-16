import h5py
import numpy as np
from keras import backend as K
from keras.applications import MobileNet, mobilenet
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils.generic_utils import CustomObjectScope

from data_generator import DataGenerator


# base model to be shared for feature extraction pretrained on MobileNet weights
def create_base_model(input_shape, num_classes):
  model = Sequential()
  # first convolution layer
  model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
                  activation='relu',
                  input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))  

  # second convolution layer
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D())
  model.add(Dropout(0.25))

  # third convolution layer
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D())
  model.add(Dropout(0.25))

  # third convolution layer
  model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(3, 3)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D())
  model.add(Dropout(0.25))

  # first fully connected layer
  model.add(Flatten())
  model.add(Dense(512, use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  # final fully connected layer
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  model.summary()
  return model

def get_data(data_path, test_data_path):
  f = h5py.File(data_path)
  x_train = np.array(f['x'].value)
  y_train = np.array(f['y'].value)
  f.close()
  num_classes = len(set(y_train))

  f = h5py.File(test_data_path)
  x_test = np.array(f['x'].value)
  y_test = np.array(f['y'].value)
  f.close()
  return (x_train, y_train), (x_test, y_test), num_classes

epochs = 15
batch_size = 8
input_shape = (224, 224, 3)
data_path = '../data/dataset.h5'
test_data_path = '../data/test_dataset.h5'
model_output_path = '../data/palm_vein_model.h5'

classes = ['angad_left', 'angad_right', 'anushka_left', 'anushka_right', 'ayush_left', 'ayush_right', 
            'cindy_left', 'cindy_right', 'david_left', 'david_right', 'edwin_left', 'edwin_right', 
            'ibrahim_left', 'ibrahim_right', 'jason_left', 'jason_right', 'jun_left', 'jun_right', 
            'justin_left', 'justin_right', 'nick_left', 'nick_right', 'samir_left', 'samir_right', 
            'thomas_left', 'thomas_right', 'will_left', 'will_right']


(x_train, y_train), (x_test, y_test), num_classes = get_data(data_path, test_data_path)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = datagen.flow(x_train, y_train)
val_generator = datagen.flow(x_test, y_test)

model = create_base_model(input_shape, num_classes)
sgd = SGD(lr=0.01, decay=1e-9, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=x_test.shape[0] // batch_size)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("[Loss, Accuracy]:", score)

'''

with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
  model = load_model('../data/palm_vein_model.h5')

  predictions = model.predict(x_test)
  for prediction in predictions:
    print(np.argmax(prediction))
    print(classes[np.argmax(prediction)])
'''
model.save(model_output_path)
