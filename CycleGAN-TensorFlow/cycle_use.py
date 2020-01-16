"""
export module
"""

import os
import sys
import cv2

MODEL_PATH='pretrained/line1162pic.pb'

IMG_SIZE=256

def inference(input_img, out_img,):
    """
    simple inference script wrap function:
    Args:
        input_img: str , input_img_path,
        out_img: str, output img path, 
    return:
        (statu, cmd)
        statu: int, speficy whether inference executed rightly,
                 0 means excuted rightly, otherwise failure 
        cmd : str , final command line scripts excuted by os
    """

    cmd = "python3 inference.py --model={0}\
                             --input={1} \
                             --output={2} \
                             --image_size={3}".format(MODEL_PATH,input_img,out_img,IMG_SIZE)
    
    return os.system(cmd),cmd

def inf_and_read(input_img):
    """
    simple inference script wrap function, given a img path, return a output img got by CycleGAN
    Args:
        input_img
    Return:
        ret_img : cv2.img ,final inference img
    """
    tmp_out = 'tmp_out.jpg'
    statu,cmd = inference(input_img,tmp_out)

    if statu != 0:
        print("runtimeError:\n cmd :{}".format(cmd))
        raise RuntimeError

    ret_img = cv2.imread(tmp_out)
    os.remove(tmp_out)
    return ret_img
        
