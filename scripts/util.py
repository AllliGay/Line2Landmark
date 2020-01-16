def thinLine(img,img_size=256):
    import numpy as np
    import cv2
    kernel = np.ones((2, 2), np.uint8)
    input_img = cv2.erode(img, kernel, iterations=1)
    ret = cv2.resize(input_img, (img_size, img_size))
    return ret
