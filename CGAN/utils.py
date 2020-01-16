"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import cv2

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

label_filename = 'datasets/landmark/boxes_split.csv'
label = {}
with open(label_filename) as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split(',')
        label[line[0]] = list(map(float, line[1].split()))
        assert len(label[line[0]]) == 4
print("get label")


def load_data(image_path, flip=True, is_test=False):
    img_A = cv2.imread(image_path)
    _, img_B = image_contour(img_A)
    # img_A, img_B = image_contour(img_A)

    image_id = image_path.split('/')[-1].split('.')[-2]
    y1, x1, y2, x2 = label.get(image_id, [0.0, 0.0, 1.0, 1.0])
    height, width, _ = img_A.shape
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    img_A = img_A[y1:y2, x1:x2]
    img_B = img_B[y1:y2, x1:x2]

    # image_con = np.concatenate((img_A, img_B), axis=1)
    # cv2.imwrite('datasets/landmark/' + image_id + '_1.jpg', img_B)

    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.
    img_B = img_B[:, :, np.newaxis]

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size,  output_c_dim + input_c_dim)
    return img_AB


def image_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    kernel = np.ones((3, 3), np.uint8)
    # edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
    edged = cv2.dilate(edged, kernel, iterations=1)
    return gray, edged


# def load_data(image_path, flip=True, is_test=False):
#     img_A, img_B = load_image(image_path)
#     img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)
#
#     img_A = img_A/127.5 - 1.
#     img_B = img_B/127.5 - 1.
#
#     img_AB = np.concatenate((img_A, img_B), axis=2)
#     # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
#     return img_AB

# def load_image(image_path):
#     input_img = imread(image_path)
#     w = int(input_img.shape[1])
#     w2 = int(w/2)
#     img_A = input_img[:, 0:w2]
#     img_B = input_img[:, w2:w]
#
#     return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = cv2.resize(img_A, (fine_size, fine_size))
        img_B = cv2.resize(img_B, (fine_size, fine_size))
    else:
        img_A = cv2.resize(img_A, (load_size, load_size))
        img_B = cv2.resize(img_B, (load_size, load_size))

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def convert_result_images(images, size):
    return (merge(inverse_transform(images), size) * 255.0).astype(np.uint8)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return cv2.imread(path, flatten = True).astype(np.float)
    else:
        return cv2.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    img = merge(images, size) * 255.0
    return cv2.imwrite(path, img)

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


