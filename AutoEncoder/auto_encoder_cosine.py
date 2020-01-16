import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os, sys, cv2
#from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyper Parameters
weight_decay = 1e-5
#weight_decay = 0
batch_size = 128
epoches = 10
model_path = '/home/zjc793532302/AE_models/'
image_size = 112
img_channels = 3

def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
        #return x

def AutoEncoder(train_x, training):
    with tf.variable_scope('AutoEncoder'):
        # encoder
        conv0 = tf.nn.relu(batch_normalization(tf.layers.conv2d(train_x, use_bias=False, filters=64, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn0'))
        conv1 = tf.nn.relu(batch_normalization(tf.layers.conv2d(conv0, use_bias=False, filters=64, \
                kernel_size=[3, 3], strides=1, padding='SAME'), training=training, scope='bn1'))
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=2, padding='SAME')
        conv2 = tf.nn.relu(batch_normalization(tf.layers.conv2d(conv1, use_bias=False, filters=128, \
                kernel_size=[3, 3], strides=1, padding='SAME'), training=training, scope='bn2'))
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=2, padding='SAME')
        conv3 = tf.nn.relu(batch_normalization(tf.layers.conv2d(conv2, use_bias=False, filters=256, \
                kernel_size=[3, 3], strides=1, padding='SAME'), training=training, scope='bn3'))
    
        feature_down = tf.reshape(conv3, [-1, 14*14*256])
        feature_down = tf.nn.relu(batch_normalization(tf.layers.dense(feature_down, 1024), \
                training=training, scope='feature_down_bn'))
        feature_out = tf.layers.dense(feature_down, 256)

        feature_up = tf.nn.relu(tf.layers.dense(feature_out, 1024))
        feature_up = tf.nn.relu(batch_normalization(tf.layers.dense(feature_up, 14*14*256), \
                training=training, scope='feature_up_bn'))
        feature_up = tf.reshape(feature_up, [-1, 14, 14, 256])
        # decoder
        deconv4 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(feature_up, use_bias=False, filters=128, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn4'))
        deconv5 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(deconv4, use_bias=False, filters=64, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn5'))
        deconv6 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(deconv5, use_bias=False, filters=32, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn6'))
        final = tf.nn.sigmoid(tf.layers.conv2d(deconv6, use_bias=False, filters=3, kernel_size=[3, 3], padding='SAME'))

    return feature_out, final

def train(learning_rate = 0.001):

    def cosine(feature1, feature2):
        len1 = tf.sqrt(tf.reduce_mean(feature1 * feature1, 1))
        len2 = tf.sqrt(tf.reduce_mean(feature2 * feature2, 1))
        mul = tf.reduce_mean(feature1 * feature2, 1)
        similarity = tf.reduce_mean(tf.div(mul, len1 * len2 + 1e-8))
        return similarity
    # tf placeholder
    input_x = tf.placeholder(tf.float32, [None, 112, 112, 3])    # value in the range of (0, 1)
    label_x = tf.placeholder(tf.float32, [None, 112, 112, 3])    # value in the range of (0, 1)
    #origin_x = tf.placeholder(tf.float32, [None, 112, 112, 3])    # value in the range of (0, 1)
    LR = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    feature_out, decoded = AutoEncoder(input_x, training)
    #feature1, decoded1 = AutoEncoder(input_x, training)
    #feature2, decoded2 = AutoEncoder(origin_x, training)
    loss = tf.losses.mean_squared_error(labels=label_x, predictions=decoded) * 100
    #loss = tf.losses.mean_squared_error(labels=input_x, predictions=decoded1) * 100 + tf.losses.mean_squared_error(labels=origin_x, predictions=decoded2) * 100
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * weight_decay
    feature1 = feature_out[::2]
    feature2 = feature_out[1::2]
    cos_loss = 1 - cosine(feature1, feature2)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss + l2_loss + cos_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    #saver = tf.train.Saver(tf.global_variables())
    #saver.restore(sess, model_path + 'AutoEncoder.ckpt')

    img_names = []
    bboxes = []
    dir_name = 'CycleGAN_data'
    files = os.listdir('/home/zjc793532302/GoogleLandmark/{}'.format(dir_name))
    for f in files:
        if '.jpg' not in f or f[:4] == 'test':
            continue
        img_names.append('/home/zjc793532302/GoogleLandmark/{}/'.format(dir_name)+f)
    
    val_imgs = img_names[-100:]
    img_names = img_names[:-100]
    #for line in lines:
    #    if os.path.exists('/home/zjc793532302/GoogleLandmark/train/train_data/{}.jpg'.format(line.strip().split(',')[0])):
    #        img_names.append('/home/zjc793532302/GoogleLandmark/train/train_data/{}.jpg'.format(line.strip().split(',')[0]))
    #        box = line.strip().split(',')[1].split()
    #        bboxes.append([float(box[i]) for i in range(4)])
    
    def get_edge(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图像
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
        edged = cv2.Canny(gray, 75, 200)  # 边缘检测
        return edged
                             
    steps = len(img_names) // batch_size
    for epoch in range(epoches):
        if epoch in [6]:
            learning_rate /= 10
        idx = 0
        #np.random.shuffle(train_x)
        import random
        for step in range(steps):
            images = []
            origin_images = []
            for i in range(batch_size):
                #print(img_path)
                img_path = img_names[idx+i]
                #box = bboxes[idx+i]
                img = cv2.imread(img_path)
                #shape = img.shape
                #img = img[int(box[0]*shape[0]):int(box[2]*shape[0]),int(box[1]*shape[1]):int(box[3]*shape[1]),:]
                img = cv2.resize(img, (image_size, image_size))
                #cv2.imwrite('results/{}.jpg'.format(i),img)
                images.append(np.array(img.reshape(1, image_size, image_size, img_channels), np.float32))
                
                origin_img = cv2.imread('/home/zjc793532302/GoogleLandmark/train/train_data/{}'.format(img_path.split('/')[-1]))
                #print('/home/zjc793532302/GoogleLandmark/test/test/{}'.format(img_path.split('/')[-1]))
                origin_img = cv2.resize(origin_img, (image_size, image_size))
                #origin_images.append(np.array(origin_img.reshape(1, image_size, image_size, img_channels), np.float32))
                images.append(np.array(origin_img.reshape(1, image_size, image_size, img_channels), np.float32))
                #image = get_edge(origin_img)
                #images.append(np.array(image.reshape(1, image_size, image_size, img_channels), np.float32))
                #print(images[0].shape)
            images = np.concatenate(images, axis=0)
            #origin_images = np.concatenate(origin_images, axis=0)
            train_data = images #cv2.imread -> BGR
            #train_data = images
            train_data = train_data / 255
            
            label_data = train_data.copy()
            label_data[::2] = train_data[1::2]
            #origin_data = origin_images[..., 0] * 0.114 + origin_images[..., 1] * 0.587 + origin_images[..., 2] * 0.299 
            #origin_data = origin_images
            #origin_data = origin_data / 255
            idx += batch_size
            
            _, mse_loss, weight_loss, sim_loss = sess.run([train_op, loss, l2_loss, cos_loss], \
                feed_dict={input_x: train_data, label_x:label_data, LR: learning_rate, training: True})
            
            if step % 1 == 0:
                val_loss = 0
                vsim_loss = 0
                for i in range(len(val_imgs) // 100):
                    images = []
                    origin_images = []
                    for img_path in val_imgs[i*100 : i*100+100]:
                        img = cv2.imread(img_path)
                        #shape = img.shape
                        #img = img[int(box[0]*shape[0]):int(box[2]*shape[0]),int(box[1]*shape[1]):int(box[3]*shape[1]),:]
                        img = cv2.resize(img, (image_size, image_size))
                        #cv2.imwrite('results/{}.jpg'.format(i),img)
                        images.append(np.array(img.reshape(1, image_size, image_size, img_channels), np.float32))
                
                        origin_img = cv2.imread('/home/zjc793532302/GoogleLandmark/train/train_data/{}'.format(img_path.split('/')[-1]))
                        origin_img = cv2.resize(origin_img, (image_size, image_size))
                        images.append(np.array(origin_img.reshape(1, image_size, image_size, img_channels), np.float32))
                        
                    images = np.concatenate(images, axis=0)
                    #origin_images = np.concatenate(origin_images, axis=0)
                    #train_data = images[..., 0] * 0.114 + images[..., 1] * 0.587 + images[..., 2] * 0.299  #cv2.imread -> BGR
                    test_data = images
                    test_data = test_data / 255
                    label_data = test_data.copy()
                    label_data[::2] = test_data[1::2]
                    #origin_data = origin_images
                    #origin_data = origin_data / 255
                    tmp_loss, val_sim_loss = sess.run([loss, cos_loss], {input_x: test_data, label_x:label_data, training: False})
                    val_loss += tmp_loss
                    vsim_loss += val_sim_loss
                val_loss /= len(val_imgs) // 100
                vsim_loss /= len(val_imgs) // 100

                #from IPython import embed
                #embed()
                print('epoch:{} step:{} mse_loss:{:.4f} l2_loss:{:.4f} cos_loss:{:.4f} val_loss:{:.4f} val_sim_loss:{:.4f}'.format(epoch, step, mse_loss, weight_loss, sim_loss, val_loss, vsim_loss))
            #if step % 1 == 0:
                #print('epoch:{} step:{} mse_loss:{:.4f} l2_loss:{:.4f} cos_loss:{:.4f}'.format(epoch, step, mse_loss, weight_loss, sim_loss))
        saver.save(sess=sess, save_path=model_path + 'AutoEncoder.ckpt')
    
    return 

def predict(image):
    # tf placeholder
    input_x = tf.placeholder(tf.float32, [None, 112, 112, 3])    # value in the range of (0, 1)
    training = tf.placeholder(tf.bool)
    feature_out, decoded = AutoEncoder(input_x, training)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, model_path + 'AutoEncoder.ckpt')
    
    data = []
    origin_data = []
    for path in test_imgs:
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        image = np.array(image.reshape(-1, image_size, image_size, img_channels),np.float32)
        #images = images[..., 0] * 0.114 + images[..., 1] * 0.587 + images[..., 2] * 0.299  #cv2.imread -> BGR
        image = image / 255
        feature, prediction = sess.run([feature_out, decoded], {input_x: image, training: False})
        #feature, prediction = predict(image)
        data.append((feature, path.split('/')[-1], np.squeeze(np.array(prediction*255, np.uint8))))
            
        origin_img = cv2.imread('/home/zjc793532302/GoogleLandmark/train/train_data/{}'.format(path.split('/')[-1]))
        origin_img = cv2.resize(origin_img, (image_size, image_size))
        origin_img = np.array(origin_img.reshape(-1, image_size, image_size, img_channels),np.float32)
        #images = images[..., 0] * 0.114 + images[..., 1] * 0.587 + images[..., 2] * 0.299  #cv2.imread -> BGR
        origin_img = origin_img / 255
        feature, prediction = sess.run([feature_out, decoded], {input_x: origin_img, training: False})
        #feature, prediction = predict(origin_img)
        origin_data.append((feature, path.split('/')[-1], np.squeeze(np.array(prediction*255, np.uint8))))
    
    return data, origin_data

def cosine(feature1, feature2):
    len1 = np.sqrt(np.mean(feature1 * feature1, 1))
    len2 = np.sqrt(np.mean(feature2 * feature2, 1))
    mul = np.mean(feature1 * feature2, 1)
    similarity = np.mean(np.divide(mul, len1 * len2 + 1e-8))
    return similarity
    
if __name__ == '__main__':

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        img_names = []
        files = os.listdir('/home/zjc793532302/GoogleLandmark/AE_data')
        for f in files:
            if '.jpg' not in f or f[:4] == 'test':
                continue
            img_names.append('/home/zjc793532302/GoogleLandmark/AE_data/'+f)
            test_imgs = img_names[-100:]
        
        data, origin_data = predict(test_imgs)
        
        result = []
        for i in range(len(data)):
            feature1, name1, iimg = data[i]
            val = -1
            name = ''
            
            for j in range(len(origin_data)):
                feature2, name2, image = origin_data[j]
                if cosine(feature1, feature2) > val:
                    val = cosine(feature1, feature2)
                    name = name2
                    img = image
            if name1 == name:
                result.append((name1, val))
                cv2.imwrite('/home/zjc793532302/results/{}'.format(name.replace('.jpg','_.jpg')), iimg)
                cv2.imwrite('/home/zjc793532302/results/{}'.format(name), img)
        print(result)
        print(len(result))
        from IPython import embed
        #embed()
        #cv2.imwrite('/home/zjc793532302/AE_results/{}'.format(sys.argv[1].split('/')[-1]),np.array(prediction.reshape(image_size, image_size, img_channels)*255, np.uint8))
    else:
        train()
