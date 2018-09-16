import numpy as np
import h5py
import cv2
import scipy.io as sio
import os

image_path = "./processed/"
dataset_path = "dataset.h5"
test_dataset_path = "test_dataset.h5"

def save_data(image_path, output_path, is_test=False):
    y = []
    x = []
    labels = []

    label_idx = -1
    for file in os.listdir(image_path):
        if is_test:
            if file.endswith("4.jpg") or file.endswith("5.jpg") or file.endswith("6.jpg") or file.endswith("7.jpg"):
                values = file.split('_')
                img = cv2.imread(image_path + file)
                img = cv2.resize(img, (224, 224))
                if values[0] + '_' + values[1] not in labels:
                    labels.append(values[0] + '_' + values[1])
                    label_idx += 1
                y.append(label_idx)
                x.append(img)
        else:
            if not (file.endswith("4.jpg") or file.endswith("5.jpg") or file.endswith("6.jpg") or file.endswith("7.jpg")):
                values = file.split('_')
                img = cv2.imread(image_path + file)
                img = cv2.resize(img, (224, 224))
                if values[0] + '_' + values[1] not in labels:
                    labels.append(values[0] + '_' + values[1])
                    label_idx += 1
                y.append(label_idx)
                x.append(img)

    x = np.array(x)
    y = np.array(y)

    try:
        os.remove(output_path)
    except OSError:
        pass

    f = h5py.File(output_path)
    f['x'] = x
    f['y'] = y
    f.close()

save_data(image_path, dataset_path, is_test=False)
save_data(image_path, test_dataset_path, is_test=True)