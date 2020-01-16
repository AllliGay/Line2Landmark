import os
import tensorflow as tf
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fine_size = 256
checkpoint_dir = './checkpoint/landmark'
input_nc = 1

def process_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def test():
    from CGAN.model import pix2pix
    test_img = cv2.imread('datasets/dfmz.png')
    test_img = process_rgb(test_img)

    test_img = cv2.resize(test_img, (fine_size,fine_size))

    # test_img = 255 - test_img
    cv2.imwrite('datasets/test_gray.jpg', test_img)
    with tf.Session() as sess:
        model = pix2pix(sess, image_size=fine_size, output_size=fine_size,
                        checkpoint_dir=checkpoint_dir, input_c_dim=input_nc)
        # result = model.predict('datasets/landmark/val/000a0a13c3a46be5.jpg')
        result = model.predict(test_img)
        cv2.imwrite('datasets/test_out.jpg',result)


if __name__ == '__main__':
    test()