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
train_data_config = {'voc_dir':"data/VOC2012",
                     'dataset':'train',
                     'randomize': True,
                     'seed': None}
params = {'num_classes': 22, 'rate': 1e-4,
          'trained_weight_path':'data/vgg16.npy',
          'save_trained_weight_path':'data/fcn3_fconv.npy',
          'predef_index':'2007_000129'} # None if not needed
	
train_dataset = dt.VOCDataSet(train_data_config)

# Hyper-parameters
batch_size = 2
iterations = 10

with tf.Session() as sess:
	# Init CNN -> load pre-trained weights from VGG16.
	vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
	batch = tf.placeholder(tf.float32, shape=[1, None, None, 3])
	label = tf.placeholder(tf.int32, shape=[None])	# lable is already vectorized before feed

	sparse_indices = tf.placeholder(tf.int64, shape=[None,2])
	sparse_values = tf.placeholder(tf.float32, shape=[None])
	sparse_bias = tf.placeholder(tf.float32, shape=[None])
	# create model and train op
        [train_op, loss] = vgg_fcn32s.train_fcn32(params=params,
                                                  image=batch,
                                                  truth=label,
						  diag_indices=sparse_indices,
                                                  diag_values=sparse_values,
                                                  add_bias=sparse_bias,
                                                  random_init_fc8=False,
                                                  save_var=True)
        trained_var_dict = vgg_fcn32s.var_dict
	print('Finished building network-fcn32.')
	init = tf.initialize_all_variables()
	sess.run(init)

	print('Start training fcn32...')
	for i in range(iterations):
		print("iter: ", i)
		for j in range(batch_size):
			next_pair = train_dataset.next_batch(params['predef_index'])		                
                        next_pair_image = next_pair[0]
                        
                        image_shape = next_pair_image.shape
                        num_pixels = image_shape[1] * image_shape[2]
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
			feed_dict = {batch: next_pair_image,
                                     label: next_pair_lable,
				     sparse_indices: sparse_indices_feed,
		       	             sparse_values: sparse_values_feed,
		                     sparse_bias: sparse_bias_feed}
                        
			sess.run(train_op, feed_dict)

			print('Loss: ', sess.run(loss, feed_dict))
	print('Finished training fcn32')


        # Save weight
        npy_path = params['save_trained_weight_path']
        weight_dict = sess.run(trained_var_dict)
        if len(weight_dict.keys()) != 0:
            for key in weight_dict.keys():
                print('Layer: %s  Weight shape: %s   Bias shape: %s'%(key, weight_dict[key][0].shape, weight_dict[key][1].shape))
                
            np.save(npy_path, weight_dict)
            print("trained weights saved: ", npy_path)
            
