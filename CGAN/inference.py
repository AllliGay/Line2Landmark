import os
import tensorflow as tf
from model import pix2pix
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fine_size = 256
batch_size = 1
checkpoint_dir = './checkpoint/landmark_128_256'
input_nc = 1


def test():
    import cv2
    import numpy as np
    test_img = cv2.imread('datasets/hqjrzx.png')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    ret, test_img = cv2.threshold(test_img, 250, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    # test_img = cv2.dilate(test_img, kernel, iterations=1)
    test_img = cv2.erode(test_img, kernel, iterations=1)
    test_img = cv2.morphologyEx(test_img, cv2.MORPH_CLOSE, kernel)
    test_img = cv2.resize(test_img, (fine_size,fine_size))

    # test_img = 255 - test_img
    cv2.imwrite('datasets/test_gray.jpg', test_img)
    with tf.Session() as sess:
        model = pix2pix(sess, image_size=fine_size, batch_size=batch_size,
            output_size=fine_size, checkpoint_dir=checkpoint_dir, input_c_dim=input_nc)
        # result = model.predict('datasets/landmark/val/000a0a13c3a46be5.jpg')
        result = model.predict(test_img)
        cv2.imwrite('datasets/test_out.jpg',result)


if __name__ == '__main__':
    test()