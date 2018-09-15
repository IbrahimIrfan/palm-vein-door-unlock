import h5py
from keras import layers, models, optimizers
from keras.applications import MobileNet

from helpers import accuracy, contrastive_loss, euclidean_dist, euclidean_dist_output_shape

epochs = 20
input_shape = (224, 224, 3)

def create_base_network(input_shape):
  # initialize pre-trained base model for fine tuning
  mobile_net_base = MobileNet(include_top=False,
                               weights='imagenet',
                               input_shape = (224, 224, 3)
                              )
  # freeze everything except last 3 layers for training
  for layer in mobile_net_base.layers[:-3]:
    layer.trainable = False

  base_model = models.Sequential()
  input = layers.Input(shape=(224, 224, 3))

  # add pre-trained model
  base_model.add(mobile_net_base)
  # add new fully-connected layers
  base_model.add(layers.Flatten())
  base_model.add(layers.Dense(1024, activation='relu'))
  base_model.add(layers.Dropout(0.1))
  base_model.add(layers.Dense(128, activation='relu'))
  base_model.add(layers.Dropout(0.1))
  base_model.add(layers.Dense(128, activation='relu'))
  return models.Model(input, base_model)

model = create_base_network(input_shape)

# Show a summary of the model. Check the number of trainable parameters
model.summary()

distance = layers.Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape)()

rms = optimizers.RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

model.save('../data/palm_vein.h5')