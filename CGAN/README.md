# CGAN for LandmarkSE
code from [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow)

## Usage
see more in `inference.py`
```python
image_size = 256
checkpoint_dir = './checkpoint/landmark'
input_nc = 1 # gray

from model import pix2pix
model = pix2pix(sess, image_size=image_size, output_size=image_size, checkpoint_dir=checkpoint_dir, input_c_dim=input_nc)
result = model.predict('xxx.jpg') # give a path
# or
result = model.predict(input_img) # give a cv2 image
```
The input should be a BINARY image. The lines are white and the background is black.
For a RGB image file whose lines are black, you can use code like this:
```python
import cv2
input_img = cv2.imread('xxx.jpg')
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
ret, input_img = cv2.threshold(input_img, 250, 255, cv2.THRESH_BINARY_INV)

import numpy as np
kernel = np.ones((5, 5), np.uint8) # or try (3, 3)?
# For whose lines are too thin, you can erode it
input_img = cv2.erode(input_img, kernel, iterations=1)
# For whose lines are too thick, you can dilate it
input_img = cv2.dilate(input_img, kernel, iterations=1)

input_img = cv2.resize(input_img, (image_size, image_size))
```
