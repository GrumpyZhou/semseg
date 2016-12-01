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

test_img1 = skimage.io.imread("./data/tabby_cat.png")

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: test_img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = FCN16VGG('data')
    preds = vgg_fcn.inference(batch_images, num_classes=20)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')
    pred32, pred16, pred8 = sess.run(preds, feed_dict=feed_dict)

    pred32_color = dt.color_image(pred32[0])
    pred16_color = dt.color_image(pred16[0])
    pred8_color = dt.color_image(pred8[0])

    scp.misc.imsave('./data/fcn32.png', pred32_color)
    scp.misc.imsave('./data/fcn16.png', pred16_color)
    scp.misc.imsave('./data/fcn8.png', pred8_color)
