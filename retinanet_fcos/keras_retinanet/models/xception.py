"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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

import keras
from keras.applications.xception import xception
from keras.utils import get_file
from ..utils.image import preprocess_image

from . import retinanet
from . import Backbone


class XceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """


    def __init__(self, backbone):
        super(XceptionBackbone, self).__init__(backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return xception_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Download pre-trained weights for the specified backbone name.
        This name is in the format mobilenet{rows}_{alpha} where rows is the
        imagenet shape dimension and 'alpha' controls the width of the network.
        For more info check the explanation from the keras mobilenet script itself.
        """
        return '/disk1/zhaojiacheng/basemodels/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        if self.backbone not in ['xception']:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackbone.allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def xception_retinanet(num_classes, backbone='xception', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    backbone = xception.Xception(input_tensor=inputs, include_top=False, pooling=None, weights=None)

    # create the full model
    layer_names = ['block3_sepconv1_act', 'block4_sepconv1_act', 'block14_sepconv2_act']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)
