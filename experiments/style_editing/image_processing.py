import cv2
import numpy as np


def increase_contrast(img):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


class Dilate(object):
    def __call__(self, img):
        return dilate(img)


class IncreaseContrast(object):
    def __call__(self, img):
        return increase_contrast(img)
