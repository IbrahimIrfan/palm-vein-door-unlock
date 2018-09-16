import h5py
import numpy as np
from keras import backend as K
from keras.applications import MobileNet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

from data_generator import DataGenerator


# base model to be shared for feature extraction pretrained on MobileNet weights
def create_base_model(input_shape, num_classes):
  # initialize pre-trained base model for fine tuning
  mobile_net_base = MobileNet(include_top=False,
                               weights='imagenet',
                               input_shape = (224, 224, 3)
                             )
  # freeze everything except last 3 layers for training
  for layer in mobile_net_base.layers[:-3]:
    layer.trainable = False

  model = Sequential()

  # add pre-trained model
  model.add(mobile_net_base)
  # add new convolution layers and fully-connected layers
  model.add(GlobalAveragePooling2D())
  model.add(Reshape((1, 1, 1024)))
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
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

epochs = 20
batch_size = 32
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
one_hot_labels = to_categorical(y_train, num_classes=num_classes)
model = create_base_model(input_shape, num_classes)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, one_hot_labels, batch_size=batch_size, epochs=epochs)

one_hot_test_labels = to_categorical(y_test, num_classes=num_classes)
score = model.evaluate(x_test, one_hot_test_labels, batch_size=batch_size)
print("Accuracy + Loss:", score)

predictions = model.predict(x_test)
print(predictions)
print(classes[np.argmax(predictions[0])])

model.save(model_output_path)