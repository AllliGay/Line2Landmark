"""
export module
"""
import tensorflow as tf
import os
import sys
import cv2
from CycleGAN import utils

IMG_SIZE=256

def inference(model_path, input_img, out_img, image_size):
    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.FastGFile(input_img, 'rb') as f:
            image_data = f.read()
            input_image = tf.image.decode_jpeg(image_data, channels=3)
            input_image = tf.image.resize_images(input_image, size=(image_size, image_size))
            input_image = utils.convert2float(input_image)
            input_image.set_shape([image_size, image_size, 3])

        with tf.gfile.FastGFile(model_path, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                                             input_map={'input_image': input_image},
                                             return_elements=['output_image:0'],
                                             name='output')

    with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        with open(out_img, 'wb') as f:
            f.write(generated)


def inf_and_read(input_img, model_path='pretrained/line1162pic.pb', image_size=IMG_SIZE):
    """
    simple inference script wrap function, given a img path, return a output img got by CycleGAN
    Args:
        input_img
    Return:
        ret_img : cv2.img ,final inference img
    """
    tmp_in = "tmp_in.jpg"
    cv2.imwrite(tmp_in, input_img)
    tmp_out = 'tmp_out.jpg'
    inference(model_path, tmp_in, tmp_out, image_size)

    ret_img = cv2.imread(tmp_out)
    os.remove(tmp_out)
    return ret_img
