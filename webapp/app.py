import sys
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf

# Some utilites
import numpy as np
import cv2
from PIL import Image
from webapp.util import base64_to_pil, np_to_base64

port = 5000
gan_img_size = 256
MODEL_NAME = 'CycleGAN' # CGAN
if MODEL_NAME is 'CGAN':
    from CGAN.inference import process_rgb
    from CGAN.model import pix2pix
    sess = tf.Session()
    gan_checkpoint_dir = 'CGAN/checkpoint/landmark'
    gan_model = pix2pix(sess, image_size=gan_img_size, output_size=gan_img_size,
                        checkpoint_dir=gan_checkpoint_dir, input_c_dim=1)
elif MODEL_NAME is 'CycleGAN':
    from CycleGAN.cycle_use import inf_and_read
    cyclegan_model_path = '../CycleGAN/pretrained/line2pic.pb'

from AutoEncoder.auto_encoder_cosine import get_sim_image

# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:' + str(port) + '/')

def gan_model_predict(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h,w = img.shape[0], img.shape[1]
    if MODEL_NAME is 'CGAN':
        img = process_rgb(img)
        img = cv2.resize(img, (gan_img_size, gan_img_size))
        generate_img = gan_model.predict(img)
    elif MODEL_NAME is 'CycleGAN':
        generate_img = inf_and_read(img, cyclegan_model_path)
    generate_img = cv2.resize(generate_img, (w, h))
    cv2.imwrite('static/test.jpg', generate_img)
    # generate_img = Image.fromarray(cv2.cvtColor(generate_img, cv2.COLOR_BGR2RGB))

    return generate_img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        generate_img = gan_model_predict(img)

        match_img = get_sim_image('static/test.jpg')

        match_img = cv2.resize(match_img, (generate_img.shape[1], generate_img.shape[0]))
        con_img = np.concatenate((generate_img, match_img), 1)
        cv2.imwrite('static/test.jpg', con_img)
        # Serialize the result, you can add additional fields
        return jsonify(result='static/test.jpg')

    return None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
