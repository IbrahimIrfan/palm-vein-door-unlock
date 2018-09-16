import os
import random

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


def get_data(base_path):
  f = h5py.File("../data/dataset.h5")
  x = f['x'].value
  y = f['y'].value
  f.close()
  x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
  return (x_train, y_train), (x_test, y_test)

def get_num_classes():
  num_classes = 0
  return num_classes

class DataGenerator(object):
  def __init__(self, batch_size, base_path):
    (x_train, y_train), (x_test, y_test) = get_data(base_path)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = get_num_classes()

    label_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    self.train_pairs, self.train_y = self.create_image_pairs(x_train, label_indices, num_classes)

    label_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    self.test_pairs, self.test_y = self.create_image_pairs(x_test, label_indices, num_classes)

    self.train_pairs_0 = self.train_pairs[:, 0]
    self.train_pairs_1 = self.train_pairs[:, 1]
    self.train_y_0 = self.train_y[:, 0]
    self.train_y_1 = self.train_y[:, 1]

    self.test_pairs_0 = self.test_pairs[:, 0]
    self.test_pairs_1 = self.test_pairs[:, 1]
    self.test_y_0 = self.test_y[:, 0]
    self.test_y_1 = self.test_y[:, 1]

    self.batch_size = batch_size
    self.samples_per_train  = (self.train_pairs.shape[0] / self.batch_size)
    self.samples_per_eval  = (self.test_pairs.shape[0] / self.batch_size)

    self.cur_train_index = 0
    self.cur_eval_index = 0

    self.train_image_gen = ImageDataGenerator(rescale = 1./255, 
                                      shear_range = 0.2, 
                                      zoom_range = 0.2,
                                      rotation_range=20,
                                      horizontal_flip = True)

    self.test_image_gen = ImageDataGenerator(rescale = 1./255)

    self.train_gen = self.create_multiple_generator(generator=self.train_image_gen,
                                              pairs_0=self.train_pairs_0,
                                              pairs_1=self.train_pairs_1,
                                              y_0=self.train_y_0,
                                              y_1=self.train_y_1,
                                              batch_size=self.batch_size)       
        
    self.test_gen = self.create_multiple_generator(generator=self.test_image_gen,
                                              pairs_0=self.test_pairs_0,
                                              pairs_1=self.test_pairs_1,
                                              y_0=self.test_y_0,
                                              y_1=self.test_y_1,
                                              batch_size=self.batch_size)       

  # creates pairs of positive and negative images for contrastive loss training
  def create_image_pairs(self, x, label_indices, num_classes):
    pairs = []
    labels = []
    min_class = min([len(label_indices[c]) for c in range(num_classes)]) - 1
    for d in range(num_classes):
      for i in range(min_class):
        z1, z2 = label_indices[d][i], label_indices[d][i + 1]
        pairs += [[x[z1], x[z2]]]
        inc = random.randrange(1, num_classes)
        dn = (d + inc) % num_classes
        z1, z2 = label_indices[d][i], label_indices[dn][i]
        pairs += [[x[z1], x[z2]]]
        labels += [1, 0]
    return np.array(pairs), np.array(labels)

  def create_multiple_generator(self, generator, pairs_0, pairs_1, y_0, y_1, batch_size):
      genX1 = generator.flow(pairs_0, y_0, batch_size=batch_size)      
      genX2 = generator.flow(pairs_1, y_1, batch_size=batch_size)      

      while True:
        X1_next = genX1.next()
        X2_next = genX2.next()
        yield [[X1_next[0], X2_next[0]]], [X1_next[1], X2_next[1]]
