import cv2, util

import os, sys

if len(sys.argv) <3:
    print("USAGE: python thin_pic.py <input_img_path> <output_img_path>")
    exit(0)

input_path = sys.argv[1]
out_path = sys.argv[2]

img = cv2.imread(input_path)
out = util.thinLine(img)
cv2.imwrite(out_path,out)
