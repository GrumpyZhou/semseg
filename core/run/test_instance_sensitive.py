'''
Testing script for fcn32 without skip architecture.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import os
import scipy as scp

import numpy as np
import tensorflow as tf

from network.instance_sensitive_fcn import InstanceSensitiveFCN8s
import data_utils as dt

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Change to Cityscape database
test_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': False,
                     'seed': None,
                     'dataset': 'train',
                     'use_box': False,
                     'use_car': True,
                     'use_person': False,
                     'pred_save_path': None,
                     'colored_save_path': None,
                     'labelIDs_save_path': None}

weight_iter = 50000
params = {'rate': 1e-5,
          'trained_weight_path':'../data/val_weights/city_fcn8s_instance_sensitive_%d.npy'%weight_iter}

test_dataset = dt.CityDataSet(test_data_config)
iterations = 46
imli = {17, 45}


with tf.Session() as sess:
    # Init model and load approriate weights-data
    inst_fcn = InstanceSensitiveFCN8s(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    obj_score, inst_score = inst_fcn.inference(image)
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Running the inference ...')
    for i in range(iterations):
        if i in imli:
            next_pair = test_dataset.next_batch()            
            next_pair_image = next_pair[0]
            print(next_pair_image.dtype)
            inf_feed_dict = {image: next_pair_image}

            score_map, feature_map = sess.run([obj_score, inst_score], feed_dict=inf_feed_dict)
            np.save('./sens/feature_im%d_%d.npy'%(i, weight_iter), feature_map)
            np.save('./sens/score_im%d_%d.npy'%(i, weight_iter), score_map)
            
            
            # Get instances from densely assembling
            prediction = inst_fcn.dense_assemble(score_map, feature_map)
            scp.misc.imsave('./sens/pred_im%d_%d.png'%(i, weight_iter), prediction)
        else:
            next_pair = test_dataset.next_batch()
            continue

        
        
