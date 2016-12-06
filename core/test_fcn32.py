'''
Testing script for fcn32 without skip architecture.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

from network.fcn_vgg16 import FCN16VGG
import data_utils as dt

# from tensorflow.python.framework import ops
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import training and validation dataset
train_data_config = {'voc_dir':"data/VOCdevkit/VOC2012",
          'dataset':'val',
          'randomize': True,
          'seed': None}
params = {'num_classes': 22,
        'load-weights': 'fcn32-semantic.npy',
        # 'load-weights': 'vgg16.npy',
        'trained-weights': None}

train_dataset = dt.VOCDataSet(train_data_config)

# hyper-parameter
num_images = 1

# test_img2 = skimage.io.imread("./data/test_img/tabby_cat.png")
test_img1 = skimage.io.imread("./data/test_img/person_bike.jpg")

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32 = FCN16VGG('data', params['load-weights'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    pred32 = vgg_fcn32.inference_fcn32(image, num_classes=params['num_classes'], random_init_fc8=False)

    print('Finished building inference network-fcn32.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference_fcn32 ...')
    for i in range(num_images):
        print("image ", i)
        # next_pair = train_dataset.next_batch()
        # feed_dict = {image: next_pair[0]}

        image_height, image_width = tf.shape(test_img1)[0], tf.shape(test_img1)[1]
        # convert to numpy integers
        image_height_val, image_width_val = image_height.eval(), image_width.eval()
        feed_image = np.reshape(test_img1, (1, image_height_val, image_width_val,3))
        feed_dict = {image: feed_image}

        preds = sess.run(pred32, feed_dict=feed_dict)

        pred32_color = dt.color_image(preds[0], num_classes=22)
        scp.misc.imsave('./data/test_img/person_bike_pred.png', pred32_color)
