import os
import random

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DataGenerator(object):
  def __init__(self, batch_size, data_path, test_data_path):
    (x_train, y_train), (x_test, y_test), num_classes = self.get_data(data_path, test_data_path)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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

  def get_data(self, data_path, test_data_path):
    f = h5py.File(data_path)
    x_train = f['x'].value
    y_train = f['y'].value
    f.close()
    num_classes = len(set(y_train))

    f = h5py.File(test_data_path)
    x_test = f['x'].value
    y_test = f['y'].value
    f.close()
    return (x_train, y_train), (x_test, y_test), num_classes

  # creates pairs of positive and negative images for contrastive loss training
  def create_image_pairs(self, x, label_indices, num_classes):
    pairs = []
    labels = []
    
    return np.array(pairs), np.array(labels)

  def create_multiple_generator(self, generator, pairs_0, pairs_1, y_0, y_1, batch_size):
      genX1 = generator.flow(pairs_0, y_0, batch_size=batch_size)
      genX2 = generator.flow(pairs_1, y_1, batch_size=batch_size)      

      while True:
        X1_next = genX1.next()
        X2_next = genX2.next()
        yield [[X1_next[0], X2_next[0]]], [X1_next[1], X2_next[1]]
