def thinLine(img):
    import numpy as np
    kernel = np.ones((5, 5), np.uint8)
    input_img = cv2.erode(input_img, kernel, iterations=1)
    return input_img