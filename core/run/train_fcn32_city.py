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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Change to Cityscape database
train_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': True,
                     'seed': None,
                     'dataset': 'train'}

val_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': True,
                     'seed': None,
                     'dataset': 'val'}

# Define the scale of the network to be trained
fcn_scale = 'fcn16s'
params = {'num_classes': 20, 'rate': 1e-4,
          'tsboard_save_path': '../tsboard_result',
          'trained_weight_path':'../data/city_fcn16_skip_new.npy',
          'save_trained_weight_path':'../data/city_%s_skip_test.npy'%fcn_scale}

# Change to Cityscape databse
train_dataset = dt.CityDataSet(train_data_config)
val_dataset = dt.CityDataSet(val_data_config)

# Hyper-parameters
train_iter = 5
validate = True
val_step = 2
#val_iter = 1

with tf.Session() as sess:
    # Init CNN -> load pre-trained weights from VGG16.
    fcn = FCN16VGG(params['trained_weight_path'])

    # Be aware of loaded data type....
    train_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    train_label = tf.placeholder(tf.int32, shape=[None])
    
    # create model and train op
    [train_op, loss] = fcn.train(params=params, image=train_img, truth=train_label, scale_min=fcn_scale, save_var=True)
    var_dict_to_train = fcn.var_dict
    tf.scalar_summary('train_loss', loss)
    
    if validate:
        val_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        val_label = tf.placeholder(tf.int32, shape=[None])
        val_loss = fcn.validate(image=val_img, truth=val_label, num_classes=params['num_classes'], scale_min=fcn_scale)
        tf.scalar_summary('val_loss', loss)
        
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(params['tsboard_save_path'], sess.graph)
    
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Start training...')
    for i in range(train_iter):
        print("train iter: ", i)
        # Load data, Already converted to BGR
        next_pair = train_dataset.next_batch()
        next_pair_image = next_pair[0]

        image_shape = next_pair_image.shape
        num_pixels = image_shape[1] * image_shape[2]
        next_pair_label = np.reshape(next_pair[1], num_pixels)	# reshape to numpy 1-D vector

        train_feed_dict = {train_img: next_pair_image,
                           train_label: next_pair_label,}
        sess.run(train_op, train_feed_dict)
        print('Training Loss: ', sess.run(loss, train_feed_dict))

        if validate and i % val_step == 0:
            #for j in range(val_iter):
             next_pair = val_dataset.next_batch()
             next_pair_image = next_pair[0]

             image_shape = next_pair_image.shape
             num_pixels = image_shape[1] * image_shape[2]
             next_pair_label = np.reshape(next_pair[1], num_pixels)	# reshape to numpy 1-D vector

             val_feed_dict = {val_img: next_pair_image,
                              val_label: next_pair_label,}
             summary, val_loss_value = sess.run([merged_summary, val_loss], val_feed_dict)
             writer.add_summary(summary, i)
             print('Validation Loss:', val_loss_value)
    print('Finished training')

    # Save weight
    npy_path = params['save_trained_weight_path']
    weight_dict = sess.run(var_dict_to_train)
    print('Saving trained weight... ')
    if len(weight_dict.keys()) != 0:
        for key in weight_dict.keys():
            print('Layer: %s  Weight shape: %s   Bias shape: %s'%(key, weight_dict[key][0].shape, weight_dict[key][1].shape))

        np.save(npy_path, weight_dict)
        print("trained weights saved: ", npy_path)

