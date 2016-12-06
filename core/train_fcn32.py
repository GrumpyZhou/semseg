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
params = {'num_classes': 22, 'rate': 1e-4,
		'load-weights': 'vgg16.npy',
		'trained-weights': 'data/fcn32-semantic.npy'}

train_dataset = dt.VOCDataSet(train_data_config)

# Hyper-parameters
batch_size = 2
iterations = 1


with tf.Session() as sess:
	# Init CNN -> load pre-trained weights from VGG16.
	vgg_fcn32s = FCN16VGG('data', params['load-weights'])
	batch = tf.placeholder(tf.float32, shape=[1, None, None, 3])
	label = tf.placeholder(tf.int32, shape=[None])	# lable is already vectorized before feed

	sparse_indices = tf.placeholder(tf.int64, shape=[None,2])
	sparse_values = tf.placeholder(tf.float32, shape=[None])
	sparse_bias = tf.placeholder(tf.float32, shape=[None])
	# create model and train op
	[train_op, loss] = vgg_fcn32s.train_fcn32(params=params, image=batch, truth=label,
										diag_indices = sparse_indices,
										diag_values = sparse_values,
										add_bias = sparse_bias)

	print('Finished building network-fcn32.')
	init = tf.initialize_all_variables()
	sess.run(init)

	print('Start training fcn32...')
	for i in range(iterations):
		print("iter: ", i)
		for j in range(batch_size):
			next_pair = train_dataset.next_batch()
			image_height, image_width = tf.shape(next_pair[0])[1], tf.shape(next_pair[0])[2]
			# convert to numpy integers
			image_height_val, image_width_val = image_height.eval(), image_width.eval()
			num_pixels = image_height_val * image_width_val

			next_pair_image = next_pair[0]	# already numpy tuple
			next_pair_lable = np.reshape(next_pair[1], num_pixels)	# reshape to numpy 1-D vector in order to extract indices

			# create values on the diagonal
			ii = np.where(next_pair_lable == 255)   # find all indices where element value is 255
			sparse_values_feed = np.ones(num_pixels, dtype=np.float32)
			np.put(sparse_values_feed, ii, [0.0])		# the values of sparse_values

			single = np.arange(num_pixels, dtype=np.int64)
			single_indices = np.reshape(single, (num_pixels,1))	# to column vector
			# to double column vectors, requested by tensorflow sparse tensor creation
			sparse_indices_feed = np.concatenate((single_indices, single_indices), axis=1)

			# create the vector to be added to the last column
			sparse_bias_feed = np.zeros(num_pixels, dtype=np.float32)
			np.put(sparse_bias_feed, ii, [1.0])

			# replace all elements with value 255 with 21 -> params['num_classes']-1.0
			np.put(next_pair_lable, ii, [params['num_classes']-1.0])

			feed_dict = {batch: next_pair_image, label: next_pair_lable,
						sparse_indices: sparse_indices_feed,
						sparse_values: sparse_values_feed,
						sparse_bias: sparse_bias_feed}
			sess.run(train_op, feed_dict)
			# print('Loss: ', loss)
	print('Finished training fcn32')
	vgg_fcn32s.save_weights(sess=sess, npy_path=params['trained-weights'])