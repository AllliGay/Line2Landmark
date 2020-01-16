# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet) [![DOI](https://zenodo.org/badge/100249425.svg)](https://zenodo.org/badge/latestdoi/100249425)

Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This architecture set based on the retinanet architecture and the intsalltion method refer to https://github.com/fizyr/keras-retinanet.

Concretely, we only use the P7 level in FPN for large building in picture and we use the anchor free ways in FCOS to replace the anchor way in retinanet.