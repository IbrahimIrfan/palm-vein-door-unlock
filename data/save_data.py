import numpy as np
import h5py
import cv2
import scipy.io as sio
import os

image_path = "./processed/"

y = []
x = []
labels = []

for file in os.listdir(image_path):
    label_idx = 0
    if file.endswith(".jpg"):
        print("reading image:" + file)
        values = file.split('_')
        img = file
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        if values[0] + '_' + values[1] not in labels:
            labels.append(values[0] + '_' + values[1])
            label_idx += 1
        y.append(label_idx)
        x.append(img)

x = np.array(x)
y = np.array(y)
f = h5py.File("dataset.h5")
f['x'] = x
f['y'] = y
f.close()