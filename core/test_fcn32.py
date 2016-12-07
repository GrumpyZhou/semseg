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
                     'dataset':'train',
                     'randomize': True,
                     'seed': None}
params = {'num_classes': 22, 'rate': 1e-4,
          'trained_weight_path':'data/fcn32s.npy'}

train_dataset = dt.VOCDataSet(train_data_config)

# hyper-parameter
num_images = 1

# test_img2 = skimage.io.imread("./data/test_img/tabby_cat.png")
test_img1 = skimage.io.imread("./data/test_img/person_bike.jpg")

with tf.Session() as sess:
    # Init model and load approriate weights-data
    vgg_fcn32s = FCN16VGG(params['trained_weight_path'])
    image = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # Build fcn32 model
    option={'fcn32s':True, 'fcn16s':False, 'fcn8s':False} 
    predict_ = vgg_fcn32s.inference(image, num_classes=params['num_classes'], random_init_fc8=False, option=option)

    print('Finished building inference network-fcn32.')
    init = tf.initialize_all_variables()
    sess.run(init)
    predict = {}

    print('Running the inference ...')
    for i in range(num_images):
        print("image ", i)
        # next_pair = train_dataset.next_batch()
        # feed_dict = {image: next_pair[0]}

        image_height, image_width = tf.shape(test_img1)[0], tf.shape(test_img1)[1]
        # convert to numpy integers
        image_height_val, image_width_val = image_height.eval(), image_width.eval()
        feed_image = np.reshape(test_img1, (1, image_height_val, image_width_val,3))
        feed_dict = {image: feed_image}
        
        predict = sess.run(predict_, feed_dict=feed_dict)
        print( len(predict))
        for key in option.keys():
            if option[key]:
                pred_color = dt.color_image(predict[key][0], num_classes=params['num_classes'])
                scp.misc.imsave('./data/test_img/person_bike_%s.png'%key, pred_color)
      
