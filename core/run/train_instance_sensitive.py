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

import os
import numpy as np
import tensorflow as tf

from network.instance_sensitive_fcn import InstanceSensitiveFCN8s

import data_utils as dt

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Change to Cityscape database
train_data_config = {'city_dir':"../data/CityDatabase",
                     #'city_dir':"/Users/WY/Downloads/CityDatabase",
                     'randomize': False,
                     'seed': None,
                     'dataset': 'train',
                     'use_box': True,
                     'use_car': True,
                     'use_person': False,
                     'pred_save_path': None,
                     'colored_save_path': None,
                     'labelIDs_save_path': None}

# Define the scale of the network to be trained
fcn_scale = 'fcn8s'

# TODO: adjust params according to instance-sensitive network
params = {'rate': 1e-6,
          'tsboard_save_path': '../data/tsboard_result/%s'%fcn_scale,
          'trained_weight_path':'../data/val_weights/fcn8s/city_fcn8s_skip_100000.npy',
          #'trained_weight_path':'../data/val_weights/city_fcn8s_skip_100000.npy',
          'save_trained_weight_path':'../data/val_weights/'}

# Change to Cityscape databse
train_dataset = dt.CityDataSet(train_data_config)

# Hyper-parameters
train_iter = 1
val_step = 1

# Logging config
with tf.Session() as sess:
    # Init CNN -> load pre-trained weights from VGG16.
    fcn = InstanceSensitiveFCN8s(params['trained_weight_path'])
    npy_path = params['save_trained_weight_path']

    # Be aware of loaded data type....
    train_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    train_box = tf.placeholder(tf.int32, shape=[256,3,2])
    train_mask = tf.placeholder(tf.int32, shape=[1, None, None, 1])

    # create model and train op
    [train_op, loss] = fcn.train(image=train_img, gt_mask=train_mask, gt_box=train_box, learning_rate=params['rate'], num_box=256, save_var=True)
    var_dict_to_train = fcn.var_dict
    #tf.scalar_summary('train_loss', loss)

    #merged_summary = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter(params['tsboard_save_path'], sess.graph)

    init = tf.initialize_all_variables()
    sess.run(init)

    print('Start training...')
    for i in range(train_iter+1):
        # Load data, Already converted to BGR
        next_pair = train_dataset.next_batch()
        next_pair_image = next_pair[0]
        next_pair_label = next_pair[1]
        next_pair_box = next_pair[2]

        image_shape = next_pair_image.shape

        train_feed_dict = {train_img: next_pair_image,
                           train_box: next_pair_box,
                           train_mask: next_pair_label}
        sess.run(train_op, train_feed_dict)
        print('loss %f'% sess.run(loss, train_feed_dict))
        # Save loss value
        # if i % 100 == 0:
        #     summary, loss_value = sess.run([merged_summary, loss], train_feed_dict)
        #     writer.add_summary(summary, i)
        #     print('Iter %d Training Loss: %f' % (i,loss_value))

        # Save weight for validation
  #       if i >= val_step and i % val_step == 0:
  #           train_weight_dict = sess.run(var_dict_to_train)
  #           print('Saving trained weight after %d iterations... '%i)
  #           if len(train_weight_dict.keys()) != 0:
  #               fname = 'city_%s_instance_sensitive_%d.npy'%(fcn_scale,i)
		# fpath = npy_path+fname
  #               np.save(fpath, train_weight_dict)
  #               print("trained weights saved: ", fpath)
    print('Finished training')



