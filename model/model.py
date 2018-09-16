import h5py
import numpy as np
from keras import backend as K
from keras.applications import MobileNet
from keras.layers import (Dense, Dropout, GlobalAveragePooling2D, Input,
                          Lambda, Reshape)
from keras.models import Model
from keras.optimizers import RMSprop

from data_generator import DataGenerator


# inspired by the MNIST siamese network with data augmentation through keras.preprocessing.image.ImageDataGenerator
# base model to be shared for feature extraction pretrained on MobileNet weights
def create_base_model(input_shape):
  # initialize pre-trained base model for fine tuning
  mobile_net_base = MobileNet(include_top=False,
                               weights='imagenet',
                               input_shape = (224, 224, 3)
                              )
  # freeze everything except last 3 layers for training
  for layer in mobile_net_base.layers[:-3]:
    layer.trainable = False

  inputs = Input(input_shape)

  # add pre-trained model
  x = mobile_net_base(inputs)
  # add new convolution layers and fully-connected layers
  x = GlobalAveragePooling2D()(x)
  x = Reshape((1, 1, 1024))(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(128, activation='relu')(x)

  base_model = Model(inputs=inputs, outputs=predictions)
  base_model.summary()
  return base_model

# euclidean distance for contrastive loss
def eucl_dist(vectors):
  x, y = vectors
  sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))

# returns output shape for euclidiean distance
def eucl_dist_output_shape(shapes):
  shape1, _ = shapes
  return (shape1[0], 1)

# contrastive loss function for one-shot learning
def contrastive_loss(y_actual, y_pred):
  margin = 1
  square_pred = K.square(y_pred)
  margin_square = K.square(K.maximum(margin - y_pred, 0))
  return K.mean(y_actual * square_pred + (1 - y_actual) * margin_square)

# keras  using contrastive loss with fixed threshold for distance
def accuracy(y_actual, y_pred):
  return K.mean(K.equal(y_actual, K.cast(y_pred < 0.5, y_actual.dtype)))

# classification accuracy with a fixed threshold on distances for Keras
def compute_accuracy(y_actual, y_pred):
  pred = y_pred.ravel() < 0.5
  return np.mean(pred == y_actual)


epochs = 20
batch_size = 32
input_shape = (224, 224, 3)
base_data_path = '../images/'

datagen = DataGenerator(batch_size, base_data_path)

base_model = create_base_model(input_shape)

# init two inputs for images
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# two different output vectors computed by the same network
feat_a = base_model(input_a)
feat_b = base_model(input_b)

# outputs euclidean distances between feature vectors
distance = Lambda(eucl_dist, output_shape=eucl_dist_output_shape)([feat_a, feat_b])

siamese_model = Model([input_a, input_b], distance)
rms = RMSprop()
siamese_model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

history = siamese_model.fit_generator(datagen.train_gen,
                                      epochs=epochs,
                                      steps_per_epoch=datagen.samples_per_train,
                                      validation_data=datagen.test_gen,
                                      validation_steps=datagen.samples_per_eval,
                                      use_multiprocessing=True,
                                      shuffle=False)

# compute final accuracy on training and test sets
y_pred = siamese_model.predict([datagen.train_pairs_0, datagen.train_pairs_1])
train_acc = compute_accuracy(datagen.train_y, y_pred)
y_pred = siamese_model.predict([datagen.test_pairs_0, datagen.test_pairs_1])
test_acc = compute_accuracy(datagen.test_y, y_pred)

print('Training accuracy:' + (100 * train_acc))
print('Test accuracy:' + (100 * test_acc))

siamese_model.save('../data/palm_vein_weights.h5')
