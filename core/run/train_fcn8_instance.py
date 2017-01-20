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

import data_utils as dt
from network.fcn_instance import InstanceFCN8s


# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Change to Cityscape database
train_data_config = {'city_dir':"../data/CityDatabase",
                     'randomize': False,
                     'use_gt_mask': True,
                     'seed': None,
                     'dataset': 'train'}

params = {'rate': 1e-4, 'num_classes': 20, 'max_instance': 30, 
          'target_class':{11:'person', 13:'car'},
          'tsboard_save_path': '../data/tsboard_result/instance',          
          'trained_weight_path':'../data/val_weights/fcn8s/city_fcn8s_skip_90000.npy',
          'save_trained_weight_path':'../data/val_weights/'}

# Load ground truth masks ##### 
train_dataset = dt.CityDataSet(train_data_config)
train_iter = 1
val_step = 1

# Logging config
print('Training config: iters %d'%train_iter)
with tf.Session() as sess:
    # Initialization
    ifcn = InstanceFCN8s(params['trained_weight_path'], params['target_class'])
    npy_path = params['save_trained_weight_path']
    train_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    train_gt_mask = tf.placeholder(tf.int32, shape=[1, None, None, len(params['target_class'])])
    
    # create model and train op    
    train_op, loss, fcn8s = ifcn.train(params=params, image=train_img, gt_masks=train_gt_mask, direct_slice=False, save_var=True)
    var_dict_to_train = ifcn.var_dict
    ##tf.scalar_summary('train_loss', loss)
    
    ##merged_summary = tf.merge_all_summaries()
    ##writer = tf.train.SummaryWriter(params['tsboard_save_path'], sess.graph)
    
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Start training...')
    for i in range(train_iter+1):
        # Load data, Already converted to BGR #####
        next_pair = train_dataset.next_batch()
        next_pair_image = next_pair[0]
        next_pair_gt_mask = next_pair[1] 
        
        train_feed_dict = {train_img: next_pair_image,
                           train_gt_mask: next_pair_gt_mask,}
        
        sess.run(train_op, train_feed_dict)
        loss_value, pred_sem = sess.run([loss, fcn8s], train_feed_dict)
        print('Iter %d Training Loss: %f' % (i,loss_value))
        imsave('../data/test_city_colored/semantic_%d.png'%i, pred_sem)

	# Save loss value
        #if i % 100 == 0:
            ##summary, loss_value = sess.run([merged_summary, loss], train_feed_dict)
            ##writer.add_summary(summary, i)
        #   print('Iter %d Training Loss: %f' % (i,loss_value))
            
        # Save weight for validation
        if i >= val_step and i % val_step == 0:
            train_weight_dict = sess.run(var_dict_to_train)
            print('Saving trained weight after %d iterations... '%i)
            if len(train_weight_dict.keys()) != 0:
                for key in train_weight_dict.keys():
                    print('Layer: %s  Weight shape: %s   Bias shape: %s'%(key, train_weight_dict[key][0].shape, train_weight_dict[key][1].shape))
                fname = 'city_instance_%d.npy'%i
		fpath = npy_path+fname
                np.save(fpath, train_weight_dict)
                print("trained weights saved: ", fpath)
    print('Finished training')

    
