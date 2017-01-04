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

from eval import evalPixelSemantic

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Import training and validation dataset
test_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': False,
                     'seed': None,
                     'dataset':'val',
                     'pred_save_path':'../data/test_city_trainIDs',
                     'colored_save_path': '../data/test_city_colored',
                     'labelIDs_save_path': '../data/test_city_labelIDs'}

params = {'num_classes': 20, 'rate': 1e-4,
          'trained_weight_path':'../data/val_weights/city_fcn16s_skip_5000.npy',
          'pred_type_prefix':'_skip_5000_'} # When saving predicting result, the prefix is
                                       # concatenated into the file name

test_dataset = dt.CityDataSet(test_data_config)
iterations = 2

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    option={'fcn32s':False, 'fcn16s':True, 'fcn8s':False}
    predict_ = vgg_fcn32s.inference(image, num_classes=params['num_classes'],
                                    scale_min='fcn16s', option=option)

    predict = {}
    accuracy = 0.0
    print('Finished building inference network-fcn16.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference ...')
    for i in range(iterations):
        print("iter:", i)
        # Load data, Already converted to BGR
        next_pair = test_dataset.next_batch()
        next_pair_image = next_pair[0]
        feed_dict = {image: next_pair_image}

        predict = sess.run(predict_, feed_dict=feed_dict)
        prefix_dict = []
        for key in option.keys():
            if option[key]:
                fname_prefix = key+params['pred_type_prefix']  # e.g fcn16_skip_ will be added into the name of pred_to_color
                prefix_dict.append(fname_prefix)
                test_dataset.save_trainID_img(fname_prefix, predict[key])
    # print("Inference done! Start transforming to colored ...")
    # test_dataset.pred_to_color()
    print("Inference done! Start transforming to labelIDs ...")
    test_dataset.pred_to_labelID(prefix_dict)
    # return averageScore over all tested images, data type: float
    # Usage: see evalPixelSemantic.py
    accuracy = evalPixelSemantic.run_eval()

