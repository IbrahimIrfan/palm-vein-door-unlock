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

# histogram equalization
def histEqualization(img);
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# invert a binary image
def invert(img):
    return cv2.bitwise_not(img)

# erosion
def erode(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    erosion = cv2.erode(gray,kernel,iterations = 1)

# skeletonize the image
def skel(img, gray):
    img = gray.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

    while cv2.countNonZero(img) != 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]

    return skel

# threshold to make the veins more visible
def thresh(img):
    ret, thr = cv2.threshold(img, 5,255, cv2.THRESH_BINARY);
    return thr

def main():
    img = cv2.imread("pic.jpg")
    kernel = np.ones((7,7),np.uint8)

    cv2.imwrite("thr.jpg", thr)

