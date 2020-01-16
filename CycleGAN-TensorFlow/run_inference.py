import os
import sys

def inference(input_img, out_img,model='pretrained/line1162pic.pb',img_size=256):
    cmd = "python3 inference.py --model={0}\
                             --input={1} \
                             --output={2} \
                             --image_size={3}".format(model,input_img,out_img,img_size)
    
    return os.system(cmd),cmd

input_dir = 'data/pic2line/train2'
out_dir = 'output116'

if not os.path.exists(input_dir):
    exit(0)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

se = set(os.listdir(out_dir))
for img in os.listdir(input_dir):
    if img.endswith('.jpg') and (img not in se):
        input_img = os.path.join(input_dir,img)
        out_img = os.path.join(out_dir,img)
        ret, cmd = inference(input_img,out_img)

        if ret !=0:
            print(input_img)
            exit(0)
