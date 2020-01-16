Keras implementation of RetinaNet object detection as described in Focal Loss for Dense Object Detection by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This architecture set based on the retinanet architecture and the intsalltion method refer to https://github.com/fizyr/keras-retinanet.

Concretely, we only use the P7 level in FPN for large building in picture and we use the anchor free ways in FCOS to replace the anchor way in retinanet.

For the predicting, you should enter the dir 'keras_retinanet/bin'.
With the order 'python predict.py {model_path} --image_path {image_path}', you can get the box of the building in the picture.
