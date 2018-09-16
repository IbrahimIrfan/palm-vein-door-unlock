import cv2
import numpy as np

# reduce noise in the image
def reduce_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

# histogram equalization
def equalize_hist(img, kernel):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# invert a binary image
def invert(img):
    return cv2.bitwise_not(img)

# gray
def gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# skeletonize the image
def skel(gray):
    img = gray.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

    while cv2.countNonZero(img) != 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]

    return skel

# threshold to make the veins more visible
def thresh(img):
    _, thr = cv2.threshold(img, 5,255, cv2.THRESH_BINARY)
    return thr

def processImage(imgIn, imgOut):
    print "processing..."
    img = cv2.imread(imgIn)
    kernel = np.ones((7,7),np.uint8)
    noise = reduce_noise(img)
    img_output = equalize_hist(noise, kernel)
    inv = invert(img_output)
    gray_scale = gray_img(inv)
    print "skeletonizing..."
    skeleton = skel(gray_scale)
    thr = thresh(skeleton)
    cv2.imwrite(imgOut, thr)
    print "done"
