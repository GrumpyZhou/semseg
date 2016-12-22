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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Import training and validation dataset
test_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': False,
                     'seed': None,
                     'dataset':'test',
                     'pred_save_path':'../data/test_city_trainIDs'}

params = {'num_classes': 20, 'rate': 1e-4,
          'trained_weight_path':'../data/city_fcn16_skip_new.npy',
          'pred_type_prefix':'_skiptest_'} # When saving predicting result, the prefix is
                                       # concatenated into the file name

test_dataset = dt.CityDataSet(test_data_config)
iterations = 3

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    option={'fcn32s':False, 'fcn16s':True, 'fcn8s':False}
    predict_ = vgg_fcn32s.inference(image, num_classes=params['num_classes'],
                                    scale_min='fcn16s', option=option)

    predict = {}
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
        for key in option.keys():
            if option[key]:
                fname_prefix = key+params['pred_type_prefix']  # e.g fcn16_skip_ will be added into the name of pred_to_color
                test_dataset.save_trainID_img(fname_prefix, predict[key])
                #test_dataset.pred_to_color(fname_prefix, predict[key])
    #print("inference done! Staring transform format for evaluation...")
    #test_dataset.convert_to_labelID(test_data_config['pred_save_path'], '../data/submit_city')

