'''
This is the first training stage -> train a network with coarse prediction vgg_fcn32s.
The trained weights will be saved into a file ".data/vgg_fcn32.npy" which will be
used in the 2nd stage training -> vgg_fcn16s and so on and so forth.
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

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import training and validation dataset
train_data_config = {'voc_dir':"data/VOCdevkit/VOC2012",
          'dataset':'train',
          'randomize': True,
          'seed': None}
params = {'num_classes': 21, 'rate': 1e-4}

train_dataset = dt.VOCDataSet(train_data_config)
# data_batch = train_dataset.next_batch()

# Hyper-parameters
batch_size = 2
iterations = 2


with tf.Session() as sess:
	# Init CNN -> load pre-trained weights from VGG16.
	vgg_fcn32s = FCN16VGG('data')
	batch = tf.placeholder(tf.float32, shape=[1, None, None, 3])
	label = tf.placeholder(tf.float32, shape=[1, 1, None, None])
	# create model and train op
	[train_op, loss] = vgg_fcn32s.train(params=params, batch=batch, label=label)

	print('Finished building network.')
	init = tf.initialize_all_variables()
	sess.run(init)

	print('Start training ...')
	for i in range(iterations):
		for j in range(batch_size):
			next_pair = train_dataset.next_batch()
			#next_pair_ = tf.reshape(next_pair[0], [1, tf.shape(next_pair[0])[2], tf.shape(next_pair[0])[3], 3])
			#next_pair__ = next_pair_.eval()		# Convert to python numpy array
                        print('image shape:', next_pair[0].shape)
                        print('label shape:', next_pair[1].shape)
                                
			feed_dict = {batch: next_pair[[0], label: next_pair[1]}
			_ = sess.run(train_op, feed_dict=feed_dict)
			print('Loss: ', loss)

	print('Finished training')
