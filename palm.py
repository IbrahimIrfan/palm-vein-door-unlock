import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# reduce noise in the image
def reduceNoise():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

# histogra equalization
def histEqualization(img);
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
