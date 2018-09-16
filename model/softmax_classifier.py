import h5py
import numpy as np
from keras import backend as K
from keras.applications import MobileNet, mobilenet
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, Reshape)
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils.generic_utils import CustomObjectScope

import tensorflowjs as tfjs
from data_generator import DataGenerator


# base model to be shared for feature extraction pretrained on MobileNet weights
def create_base_model(input_shape, num_classes):

  # initialize pre-trained base model for fine tuning
  mobile_net_base = MobileNet(include_top=False,
                               weights='imagenet',
                               input_shape = (224, 224, 3)
                              )

  # freeze everything except last 3 layers for training
  for layer in mobile_net_base.layers[:-6]:
    layer.trainable = False

  model = Sequential()

  # add pre-trained model
  model.add(mobile_net_base)
  model.add(GlobalAveragePooling2D())
  model.add(Reshape((1, 1, 1024)))
  model.add(Flatten())

  # fully connected layers
  model.add(Dense(256, use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(128, use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  # final fully connected prediction layer
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

epochs = 10
batch_size = 16
input_shape = (224, 224, 3)
data_path = '../data/dataset.h5'
test_data_path = '../data/test_dataset.h5'
model_output_path = '../data/palm_vein_model.h5'
tfjs_target_dir = '../web/model/'

classes = ['ayush_left', 'ayush_right', 'ibrahim_left', 'ibrahim_right']

(x_train, y_train), (x_test, y_test), num_classes = get_data(data_path, test_data_path)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

'''
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
'''
model = load_model(model_output_path)
#model = create_base_model(input_shape, num_classes)

sgd = SGD(lr=0.0007, decay=1e-6, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
'''
model.fit_generator(train_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=x_test.shape[0] // batch_size)
'''

score = model.evaluate(x_test, y_test)
print("[Loss, Accuracy]:", score)

predictions = model.predict(x_test)
for prediction in predictions:
  print(np.argmax(prediction), prediction)
  print(classes[np.argmax(prediction)])

model.save(model_output_path)
tfjs.converters.save_keras_model(model, tfjs_target_dir)
