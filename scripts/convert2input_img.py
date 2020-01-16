import cv2

import sys

if len(sys.argv) <3:
    print("USAGE: python convert2input_img.py  <input_img_path> <output_img_path>")

img_path = sys.argv[1]
out_path = sys.argv[2]

input_img = cv2.imread(img_path)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
ret, input_img = cv2.threshold(input_img, 250, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite(out_path,input_img)
