'''
Testing script for fcn32 without skip architecture.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import os
import numpy as np
import tensorflow as tf

from network.fcn_instance import InstanceFCN8s
import data_utils as dt

from scipy.misc import toimage
from scipy.misc import imsave

from eval import evalPixelSemantic

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import training and validation dataset
test_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': False,
                     'use_gt_mask': True,
                     'seed': None,
                     'dataset':'val',
                     'pred_save_path':'../data/test_city_trainIDs',
                     'colored_save_path': '../data/test_city_colored',
                     'labelIDs_save_path': '../data/test_city_labelIDs'}

params = {'num_classes': 20, 'max_instance': 20, 
          'target_class':{11:'person', 13:'car'},
          'trained_weight_path':'../data/val_weights/city_instance_80000.npy'}

test_dataset = dt.CityDataSet(test_data_config)
iterations = 5


with tf.Session() as sess:
    # Initialization
    ifcn = InstanceFCN8s(params['trained_weight_path'], params['target_class'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn8s_instance, return masks of each class [mask_11,mask_13]
    # each mask has shape [1, h, w, 1]
    predict = ifcn.inference(params, image, direct_slice=False)
    print('Finished building inference network-fcn8s_instance.')
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference ...')
    for i in range(iterations):
        # Load data, Already converted to BGR
        next_pair = test_dataset.next_batch()
        next_pair_image = next_pair[0]
        feed_dict = {image: next_pair_image}
        
        predict_ = sess.run(predict, feed_dict=feed_dict)
        #imsave('../data/test_city_instance/person_%d.png'%i,predict_[0])
        #imsave('../data/test_city_instance/car_%d.png'%i, predict_[1])
        pname = '../data/test_city_instance/person_%d.png'%i
        cname = '../data/test_city_instance/car_%d.png'%i
        toimage(predict_[0], high=params['max_instance'], low=0, cmin=0, cmax=params['max_instance']).save(pname)
        toimage(predict_[1], high=params['max_instance'], low=0, cmin=0, cmax=params['max_instance']).save(cname)
