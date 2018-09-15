import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def reduceNoise():
     # noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
