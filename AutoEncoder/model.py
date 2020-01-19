import os
import tensorflow as tf
import cv2
import numpy as np
from AutoEncoder.auto_encoder_cosine import AutoEncoder

image_size = 112
img_channels = 3


def cosine(feature1, feature2):
    len1 = np.sqrt(np.mean(feature1 * feature1, 1))
    len2 = np.sqrt(np.mean(feature2 * feature2, 1))
    mul = np.mean(feature1 * feature2, 1)
    similarity = np.mean(np.divide(mul, len1 * len2 + 1e-8))
    return similarity


class model(object):
    def __init__(self, sess, model_path='../AutoEncoder'):
        self.sess = sess
        # tf placeholder
        self.input_x = tf.placeholder(tf.float32, [None, 112, 112, 3])  # value in the range of (0, 1)
        self.training = tf.placeholder(tf.bool)
        self.feature_out, self.decoded = AutoEncoder(self.input_x, self.training)

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, '../AutoEncoder/AutoEncoder.ckpt')

        self.boxes = {}
        with open('../AutoEncoder/bboxes.csv') as f:
            lines = f.readlines()
        for line in lines:
            info = line.strip().split(',')
            self.boxes[info[0]] = info[1].split()

    def predict(self, test_imgs, test_img=None):
        data = []
        origin_data = []
        for path in test_imgs:
            image = cv2.imread(path)
            image = cv2.resize(image, (image_size, image_size))
            image = np.array(image.reshape(-1, image_size, image_size, img_channels), np.float32)
            image = image / 255
            feature, prediction = self.sess.run([self.feature_out, self.decoded], {self.input_x: image, self.training: False})
            data.append((feature, path.split('/')[-1], np.squeeze(np.array(prediction * 255, np.uint8))))

            origin_img = cv2.imread('../AutoEncoder/origin_img/{}'.format(path.split('/')[-1]))
            box = self.boxes[path.split('/')[-1][:-4]]
            shape = origin_img.shape
            origin_img = origin_img[int(shape[0] * float(box[0])):int(shape[0] * float(box[2])), \
                         int(shape[1] * float(box[1])):int(shape[1] * float(box[3]))]

            origin_img = cv2.resize(origin_img, (image_size, image_size))
            origin_img = np.array(origin_img.reshape(-1, image_size, image_size, img_channels), np.float32)
            origin_img = origin_img / 255
            feature, prediction = self.sess.run([self.feature_out, self.decoded], {self.input_x: origin_img, self.training: False})

            origin_data.append((feature, path.split('/')[-1], np.squeeze(np.array(prediction * 255, np.uint8))))

        if test_img != None:
            image = cv2.imread(test_img)
            image = cv2.resize(image, (image_size, image_size))
            image = np.array(image.reshape(-1, image_size, image_size, img_channels), np.float32)
            image = image / 255
            feature, prediction = self.sess.run([self.feature_out, self.decoded], {self.input_x: image, self.training: False})

            return data, origin_data, feature

        return data, origin_data

    def get_sim_image(self, image_path):
        img_names = []
        files = os.listdir('../AutoEncoder/origin_img/')
        for f in files:
            if '.jpg' not in f or f[:4] == 'test':
                continue
            img_names.append('../AutoEncoder/origin_img/' + f)

        data, origin_data, feature = self.predict(img_names, image_path)

        val = -1
        name = ''
        for i in range(len(origin_data)):
            feature1, name1, image = origin_data[i]
            cos_val = cosine(feature1, feature)
            # print(name1, cos_val)
            if cos_val > val:
                val = cos_val
                name = name1

        print('Find the similar image with the name:{} val:{}'.format(name, val))
        image = cv2.imread('../AutoEncoder/origin_img/' + name)
        box = self.boxes[name[:-4]]
        shape = image.shape
        image = image[int(shape[0] * float(box[0])):int(shape[0] * float(box[2])), \
                int(shape[1] * float(box[1])):int(shape[1] * float(box[3]))]
        return image