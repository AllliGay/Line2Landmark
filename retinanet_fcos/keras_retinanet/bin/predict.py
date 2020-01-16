#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os,cv2
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
#from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
import numpy as np
from PIL import Image
#import progressbar
#assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
from ..utils.image import resize_image


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class pascal_voc(keras.utils.Sequence):
    def __init__(self, image_set='test', year='0712'):
        self._year = year
        self._image_set = image_set
        self._data_path = '/disk1/zhaojiacheng/VOC'+year
        self._classes = [ 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._size = len(self._image_index)
        # Default to roidb handler
        self._comp_id = 'comp3'
        self.result_path = os.path.join('/home/zhaojiacheng/swap','results','VOC2012',
            'Main',self._comp_id + '_det_' + self._image_set + '_{:s}.txt')

        # PASCAL specific config options
        self.config = {'cleanup'     : False,
                       'use_salt'    : False,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def read_image_bgr(self, path):
        """ Read an image in BGR format.

        Args
            path: Path to the image.
        """
        image = np.asarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1].copy()

    def preprocess_image(self, x, mode='caffe'):
        """ Preprocess an image by subtracting the ImageNet mean.

        Args
            x: np.array of shape (None, None, 3) or (3, None, None).
            mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.

        Returns
            The input with the ImageNet mean subtracted.
        """
        x = x.astype(np.float32)

        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

        return x

    def load_image(self, index):
        path = self.image_path_at(index)
        image = self.read_image_bgr(path)
        return image

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self.result_path.format(cls)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[im_ind][cls_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def __next__(self):
        for index in self._image_index:
            image = read_image_bgr(index)
            image = preprocess_image(image)
            yield image

def _get_detections(model, args):
    #all_detections = [[None for i in range(generator._num_classes)] for j in range(generator._size)]
    with open('/home/zjc793532302/GoogleLandmark/clean_test.txt') as f:
        lines = f.readlines()
    print(len(lines))
    #for i in progressbar.progressbar(range(generator._size), prefix='Running network: '):
    for i, line in enumerate(lines):
        #if i not in [25,27,29,4] or i < 30:
        #    continue
        path = '/home/zjc793532302/GoogleLandmark/test/test/{}'.format(line.strip())
        if os.path.exists(path) == 0:
            continue
        #raw_image    = generator.load_image(i)
        #path = '1.jpg'
        image = np.asarray(Image.open(path).convert('RGB'))
        raw_image = image[:, :, ::-1].copy()
        mode = 'caffe'
        if 'mobilenet' in args.backbone or 'densenet' in args.backbone or 'xception' in args.backbone:
            mode = 'tf'
        
        image = np.array(raw_image.copy(), np.float32)
        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68
        #image        = generator.preprocess_image(raw_image.copy(), mode)
        image, scale = resize_image(image, args.image_min_side, args.image_max_side)
        raw_image, scale = resize_image(raw_image, args.image_min_side, args.image_max_side)
        
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        # correct boxes for image scale
        #boxes /= scale
        
        from IPython import embed
        #embed()
        h, w, _ = raw_image.shape
        feature_h = (h + 127) // 128
        feature_w = (w + 127) // 128
        
        #index = scores[0].argmax(0)
        indexes = np.argsort(-scores.squeeze())[:1]
        from IPython import embed
        #embed()
        for index in indexes:
            
            box = boxes[0][index]
            from IPython import embed
            #embed()
            y = index // feature_w
            x = index - y * feature_w
        
            cv2.rectangle(raw_image, (int((x-box[0])*128), int((y-box[1])*128)),\
                      (int((x+box[2])*128), int((y+box[3])*128)),(255,0,0),3)
            
            with open('test_boxes.txt','a') as f:
                f.write('{},{} {} {} {}\n'.format(path.split('/')[-1],int((x-box[0])*128/scale),\
                      int((y-box[1])*128/scale), int((x+box[2])*128/scale), int((y+box[3])*128/scale)))
                                           
        cv2.imwrite('/home/zjc793532302/retinanet/results/{}.jpg'.format(i), raw_image)

def _get_single_detection(model, path):
    #path = args.image_path
    if os.path.exists(path) == 0:
        print('image path not exists')
        return
    image = np.asarray(Image.open(path).convert('RGB'))
    raw_image = image[:, :, ::-1].copy()
    mode = 'caffe'
    if 'mobilenet' in args.backbone or 'densenet' in args.backbone or 'xception' in args.backbone:
        mode = 'tf'
        
    image = np.array(raw_image.copy(), np.float32)
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    image, scale = resize_image(image, args.image_min_side, args.image_max_side)
    raw_image, scale = resize_image(raw_image, args.image_min_side, args.image_max_side)
        
    if keras.backend.image_data_format() == 'channels_first':
        image = image.transpose((2, 0, 1))

    # run network
    boxes, scores = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        
    from IPython import embed
    #embed()
    h, w, _ = raw_image.shape
    feature_h = (h + 127) // 128
    feature_w = (w + 127) // 128
        
    #index = scores[0].argmax(0)
    indexes = np.argsort(-scores.squeeze())[:1]
    for index in indexes:
            
        box = boxes[0][index]
        y = index // feature_w
        x = index - y * feature_w
        
        cv2.rectangle(raw_image, (int((x-box[0])*128), int((y-box[1])*128)),\
                      (int((x+box[2])*128), int((y+box[3])*128)),(255,0,0),3)
            
        box = [int((x-box[0])*128/scale),int((y-box[1])*128/scale),int((x+box[2])*128/scale), int((y+box[3])*128/scale)]
                                           
        cv2.imwrite('/home/zjc793532302/retinanet/results/{}.jpg'.format(0), raw_image)
    return  box

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--image_path',       help='Path to the test image.', default=None)
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true', default=True)
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', default='-1')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).',default='/home/zjc793532302/results/')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=600)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=800)

    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    #generator = pascal_voc()

    
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model)
    '''
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=None)
    '''
    # print model summary
    # print(model.summary())
    
    if args.image_path != None:
        box = _get_single_detection(model, args.image_path)
    else:
        # start evaluation
        all_boxes = _get_detections(
        
            model,
            args
        )


if __name__ == '__main__':
    main()
