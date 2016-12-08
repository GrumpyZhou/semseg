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
train_data_config = {'voc_dir':"data/VOC2012",
                     'dataset':'val',
                     'classes':['person'],# All classes are loaded if class is None
                     'filter_no_label':False,# Filter all indices with no ground truth,
                                             # Set to True only when you know it will happen, 
                                             #e.g you defined a class
                                             # Default is false
                     'randomize': True,
                     'seed': None}

params = {'num_classes': 22, 'rate': 1e-4,
          'trained_weight_path':'data/train_bird.npy'}

val_dataset = dt.VOCDataSet(train_data_config)
iterations = 3

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    option={'fcn32s':True, 'fcn16s':False, 'fcn8s':False} 
    predict_ = vgg_fcn32s.inference(image, num_classes=params['num_classes'], random_init_fc8=False, option=option)

    predict = {}    
    print('Finished building inference network-fcn32.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference ...')
    for i in range(iterations):
        print("iter:", i)
        next_pair = val_dataset.next_batch()
        idx = val_dataset.indices[val_dataset.idx]

        next_pair_image = next_pair[0]
        feed_dict = {image: next_pair_image}

        predict = sess.run(predict_, feed_dict=feed_dict)
        for key in option.keys():
            if option[key]:
                pred_color = dt.color_image(predict[key][0], num_classes=params['num_classes'])
                img_fpath = './data/test_img/%s_%s_%s.png'%(train_data_config['classes'][0],key,idx)
                scp.misc.imsave(img_fpath, pred_color)
                print('Image saved: %s'%img_fpath)
      
