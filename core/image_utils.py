import numpy as np
import cv2

def resize_image(image, size):
    image = image.resize(size)
    return np.array(image)

def normalize_image(image):
    return image/128 - 1 

def preproccess_image(image, size, normalize=True):
    image = resize_image(image, size)
    if(normalize):
        image = normalize_image(image)
    image = np.moveaxis(image, -1, 0)
    return image


def process_image(image, size = (384, 384)):
    image = cv2.resize(image, size).astype(np.float32)
    image = image/128 -1
    image = np.moveaxis(image, -1, 0)
    return image