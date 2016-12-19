'''
This is the first training stage -> train a network with coarse prediction vgg_fcn32s.
The trained weights will be saved into a file ".data/vgg_fcn32.npy" which will be
used in the 2nd stage training -> vgg_fcn16s and so on and so forth.
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

import numpy as np
import tensorflow as tf

from network.fcn_vgg16 import FCN16VGG
import data_utils as dt

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Import training and validation dataset
# Change to Cityscape database
train_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': True,
                     'seed': None,
                     'dataset': 'train'}

params = {'num_classes': 20, 'rate': 1e-4,
          'trained_weight_path':'../data/city_fcn32.npy',
          'save_trained_weight_path':'../data/city_fcn16_skip.npy'}

# Change to Cityscape databse
train_dataset = dt.CityDataSet(train_data_config)

# Hyper-parameters
iterations = 5000

with tf.Session() as sess:
    # Init CNN -> load pre-trained weights from VGG16.
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])

    # Be aware of loaded data type....
    batch = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    label = tf.placeholder(tf.int32, shape=[None])	# label is already vectorized before feed

    # create model and train op
    [train_op, loss] = vgg_fcn32s.train_fcn16(params=params,
                                              image=batch,
                                              truth=label,
                                              random_init_fc8=False,
                                              save_var=True)
    trained_var_dict = vgg_fcn32s.var_dict
    print('Finished building network-fcn16-skip.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Start training cn16-skip...')
    for i in range(iterations):
        print("iter: ", i)
        # Load data, Already converted to BGR
        next_pair = train_dataset.next_batch()
        next_pair_image = next_pair[0]

        image_shape = next_pair_image.shape
        num_pixels = image_shape[1] * image_shape[2]
        next_pair_label = np.reshape(next_pair[1], num_pixels)	# reshape to numpy 1-D vector

        feed_dict = {batch: next_pair_image,
                     label: next_pair_label,}

        sess.run(train_op, feed_dict)

        print('Loss: ', sess.run(loss, feed_dict))
    print('Finished training cn16-skip')


    # Save weight
    npy_path = params['save_trained_weight_path']
    weight_dict = sess.run(trained_var_dict)
    if len(weight_dict.keys()) != 0:
        for key in weight_dict.keys():
            print('Layer: %s  Weight shape: %s   Bias shape: %s'%(key, weight_dict[key][0].shape, weight_dict[key][1].shape))

        np.save(npy_path, weight_dict)
        print("trained weights saved: ", npy_path)

