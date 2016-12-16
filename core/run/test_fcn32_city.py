'''
Testing script for fcn32 without skip architecture.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc
from PIL import Image

import numpy as np
import tensorflow as tf

from network.fcn_vgg16 import FCN16VGG
import data_utils as dt
import glob

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import training and validation dataset
train_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': True,
                     'seed': None}

params = {'num_classes': 20, 'rate': 1e-4,
          'trained_weight_path':'../data/city_fcn32.npy'}

val_dataset = dt.CityDataSet(train_data_config)

# a simple test image, reshape to [1,H,W,3]
test_image_files = glob.glob('../data/test_train/image*.png')

iterations = 1

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    option={'fcn32s':True, 'fcn16s':True, 'fcn8s':True}
    predict_ = vgg_fcn32s.inference(image, num_classes=params['num_classes'], random_init_fc8=False, option=option)

    predict = {}
    print('Finished building inference network-fcn32.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference ...')
    for i in range(iterations):
        print("iter:", i)
        # IMPORTANT: if use next_batch() to fetch image, then image already in BGR order,
        # inference will convert again, and makes it RGB!
        # if use a single test image, then image is in RGB -> Ok!
        # next_pair = val_dataset.next_batch()
        # idx = val_dataset.indices[val_dataset.idx]

        # next_pair_image = next_pair[0]
        test_file = test_image_files[i]
        test_img = Image.open(test_file)
        test_img = np.array(test_img, dtype=np.float32)
        test_img = test_img[np.newaxis, ...]

        next_pair_image = test_img
        feed_dict = {image: next_pair_image}

        predict = sess.run(predict_, feed_dict=feed_dict)
        #img_fpath = test_file.replace('image', 'colored')
        for key in option.keys():
            if option[key]:
		img_fpath = test_file.replace('image', 'colored' + key)
                val_dataset.pred_to_color(img_fpath, predict[key])
                # pred_color = dt.color_image(predict[key][0], num_classes=params['num_classes'])
                # img_fpath = './data/test_img/%s_%s_%s.png'%(train_data_config['classes'][0],key,idx)
                # scp.misc.imsave(img_fpath, pred_color)
                # print('Image saved: %s'%img_fpath)

