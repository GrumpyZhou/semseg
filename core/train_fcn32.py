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
params = {'num_classes': 22, 'rate': 1e-4}

train_dataset = dt.VOCDataSet(train_data_config)
# data_batch = train_dataset.next_batch()

# Hyper-parameters
batch_size = 2
iterations = 2


with tf.Session() as sess:
	# Init CNN -> load pre-trained weights from VGG16.
	vgg_fcn32s = FCN16VGG('data')
	batch = tf.placeholder(tf.float32, shape=[1, None, None, 3])
	label = tf.placeholder(tf.int32, shape=[None])	# lable is already vectorized before feed

	# single_indices = tf.placeholder(tf.int64, shape=[None])
	num_pixels = tf.placeholder(tf.int32, shape=[1])
	sparse_values = tf.placeholder(tf.float32, shape=[None])
	sparse_bias = tf.placeholder(tf.int32, shape=[None])
	# create model and train op
	[train_op, loss] = vgg_fcn32s.train(params=params, batch=batch, label=label,
										#single_indices = single_indices,
										num_pixels = num_pixels,
										sparse_values = sparse_values,
										add_bias = sparse_bias)

	print('Finished building network.')
	init = tf.initialize_all_variables()
	sess.run(init)

	print('Start training ...')
	for i in range(iterations):
		for j in range(batch_size):
			next_pair = train_dataset.next_batch()
			image_height, image_width = tf.shape(next_pair[0])[1], tf.shape(next_pair[0])[2]
			# convert to numpy integers
			image_height_val, image_width_val = image_height.eval(), image_width.eval()
			num_pixels = image_height_val * image_width_val

			# next_pair_image = tf.reshape(next_pair[0], [1, tf.shape(next_pair[0])[1], tf.shape(next_pair[0])[2], 3])
			next_pair_image = next_pair[0]	# already numpy tuple
			next_pair_lable = np.reshape(next_pair[1], num_pixels)	# reshape to numpy 1-D vector in order to extract indices
			# next_pair_lable_ = np.reshape(next_pair[1], (num_pixels,1))	# reshape to column vector to feed_dict
			# next_pair_image_ = next_pair_image.eval()		# Convert to python numpy array
			# next_pair_lable_ = next_pair_lable.eval()		# Convert to python numpy array

			# create matrix (sparse square) to be left multiplied to the prediction matrix
			ii = np.where(next_pair_lable == 255)   # find all indices where element value is 255
			xx = np.ones(num_pixels)
			np.put(xx, ii, [0])		# xx is the values of sparseTensor
			sparse_values = tf.cast(xx, tf.float32)

			# single_indices = np.arange(num_pixels, dtype=np.int64)
			# single_indices_ = tf.cast(single_indices, tf.int64)
			# single_indices = np.reshape(single, num_pixels)
			# single_indices = tf.constant(single, dtype=tf.int64, shape=[num_pixels, 1])
			# single_indices = np.reshape(single, (num_pixels,1))
			# sparse_indices = np.concatenate((single_column, single_column), axis=1)
			# sparse_indices_ = tf.constant(sparse_indices, dtype=tf.int64, shape=[num_pixels,2])
			# spare_diag_matrix = tf.SparseTensor(sparse_indices, sparse_values, [num_pixels, num_pixels])
			# spare_diag_matrix = scp.sparse.dia_matrix((xx, [0]), shape=(num_pixels, num_pixels))

			# create the vector to be added to the last column
			yy = np.zeros(num_pixels)
			np.put(yy, ii, [1])

			feed_dict = {batch: next_pair_image, label: next_pair_lable,
						#single_indices: single_indices_,
						num_pixels: num_pixels,
						sparse_values: sparse_values,
						sparse_bias: yy}
			_ = sess.run(train_op, feed_dict=feed_dict)
			print('Loss: ', loss)

	print('Finished training')