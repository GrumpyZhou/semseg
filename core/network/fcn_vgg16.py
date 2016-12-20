"""Compact interfaces lib for a neural network including:
-- Interfaces to define a nn layer e.g conv, pooling, relu, fcn, dropout etc
-- Interfaces for variable initialization
-- Interfaces for network data post-processing e.g logging, visualizing and so on
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import nn
import data_utils as dt

DATA_DIR = 'data'

class FCN16VGG:

    def __init__(self, data_path=None):
        # Load pretrained weight
        data_dict = dt.load_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}

    def _build_model(self, image, num_classes, is_train=False, save_var=False):
        model = {}
        feed_dict = self.data_dict

        if save_var:
            var_dict = self.var_dict
        else:
            var_dict = None

        model['conv1_1'] = nn.conv_layer(image, feed_dict, "conv1_1", var_dict=var_dict)
        model['conv1_2'] = nn.conv_layer(model['conv1_1'], feed_dict, "conv1_2", var_dict=var_dict)
        model['pool1'] = nn.max_pool_layer(model['conv1_2'], "pool1")

        model['conv2_1'] = nn.conv_layer(model['pool1'], feed_dict, "conv2_1", var_dict=var_dict)
        model['conv2_2'] = nn.conv_layer(model['conv2_1'], feed_dict, "conv2_2", var_dict=var_dict)
        model['pool2'] = nn.max_pool_layer(model['conv2_2'], "pool2")

        model['conv3_1'] = nn.conv_layer(model['pool2'], feed_dict, "conv3_1", var_dict=var_dict)
        model['conv3_2'] = nn.conv_layer(model['conv3_1'], feed_dict, "conv3_2", var_dict=var_dict)
        model['conv3_3'] = nn.conv_layer(model['conv3_2'], feed_dict, "conv3_3", var_dict=var_dict)
        model['pool3'] = nn.max_pool_layer(model['conv3_3'], "pool3")

        model['conv4_1'] = nn.conv_layer(model['pool3'], feed_dict, "conv4_1", var_dict=var_dict)
        model['conv4_2'] = nn.conv_layer(model['conv4_1'], feed_dict, "conv4_2", var_dict=var_dict)
        model['conv4_3'] = nn.conv_layer(model['conv4_2'], feed_dict, "conv4_3", var_dict=var_dict)
        model['pool4'] = nn.max_pool_layer(model['conv4_3'], "pool4")


        model['conv5_1'] = nn.conv_layer(model['pool4'], feed_dict, "conv5_1", var_dict=var_dict)
        model['conv5_2'] = nn.conv_layer(model['conv5_1'], feed_dict, "conv5_2", var_dict=var_dict)
        model['conv5_3'] = nn.conv_layer(model['conv5_2'], feed_dict, "conv5_3", var_dict=var_dict)
        model['pool5'] = nn.max_pool_layer(model['conv5_3'], "pool5")

        model['conv6_1'] = nn.conv_layer(model['pool5'], feed_dict, "conv6_1", shape=[3, 3, 512, 512], dropout=is_train, keep_prob=0.5, var_dict=var_dict)
        model['conv6_2'] = nn.conv_layer(model['conv6_1'], feed_dict, "conv6_2", shape=[3, 3, 512, 512], dropout=is_train, keep_prob=0.5, var_dict=var_dict)
        model['conv6_3'] = nn.conv_layer(model['conv6_2'], feed_dict, "conv6_3", shape=[3, 3, 512, 4096], dropout=is_train, keep_prob=0.5, var_dict=var_dict)
        model['conv7'] = nn.conv_layer(model['conv6_3'], feed_dict, "conv7", shape=[1, 1, 4096, 4096], dropout=is_train, keep_prob=0.5, var_dict=var_dict)

        model['score_fr'] = nn.conv_layer(model['conv7'], feed_dict, "score_fr", shape=[1, 1, 4096, num_classes], relu=False, dropout=False, var_dict=var_dict)
        

        return model

    def inference(self, image, num_classes,option={'fcn32s':True, 'fcn16s':False, 'fcn8s':False}):
        # Image preprocess: RGB -> BGR
        # red, green, blue = tf.split(3, 3, image)
        # image = tf.concat(3, [blue, green, red])

        # Basic model
        model = self._build_model(image, num_classes, is_train=False)

        predict = {}

        # FCN-32s
        if option['fcn8s'] or option['fcn16s'] or option['fcn32s']:
            upscore32 = nn.upscore_layer(model['score_fr'],
                                         "upscore32",
                                         tf.shape(image),
                                         num_classes,
                                         ksize=64, stride=32)
            predict['fcn32s'] =  tf.argmax(upscore32, dimension=3)

        # FCN-16s
        if option['fcn8s'] or option['fcn16s']:
            upscore2_fr = nn.upscore_layer(model['score_fr'],
                                           "upscore2_fr",
                                           tf.shape(model['pool4']),
                                           num_classes,
                                           ksize=4, stride=2)

            # Fuse fc8 *2, pool4
            in_features = model['pool4'].get_shape()[3].value
            score_pool4 = nn.conv_layer(model['pool4'], self.data_dict, "score_pool4", shape=[1, 1, in_features, num_classes], relu=False, dropout=False, var_dict=self.var_dict)
        
            fuse_pool4 = tf.add(upscore2_fr, score_pool4)

            # Upsample fusion *16
            upscore16 = nn.upscore_layer(fuse_pool4,
                                         "upscore16",
                                         tf.shape(image),
                                         num_classes,
                                         ksize=32, stride=16)
            predict['fcn16s'] = tf.argmax(upscore16, dimension=3)

        # FCN-8s
        if option['fcn8s']:
            # Upsample fc8 *4
            upscore4_fr = nn.upscore_layer(fuse_pool4,    # output from last layer
                                           "upscore4_fr",
                                           tf.shape(model['pool3']),   # reshape to output of pool3
                                           num_classes,
                                           ksize=4, stride=2)

            # Fuse fc8 *4, pool4 *2, pool3            
            in_features = model['pool3'].get_shape()[3].value
            score_pool3 = nn.conv_layer(model['pool3'], self.data_dict, "score_pool4", shape=[1, 1, in_features, num_classes], relu=False, dropout=False, var_dict=self.var_dict)

            fuse_pool3 = tf.add(score_pool3, upscore4_fr)

            # # Upsample fusion *8
            upscore8 = nn.upscore_layer(fuse_pool3,
                                        "upscore8",
                                        tf.shape(image),    # reshape to original input image size
                                        num_classes,
                                        ksize=16, stride=8)

            predict['fcn8s'] = tf.argmax(upscore8, dimension=3)

        return predict

    # train model with an accuracy of 32-stride
    def train_fcn32(self, params, image, truth, save_var=False):

        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        '''

        # Important: When training, random_init_fc8=True. When inference, random_init_fc8=False
        model = self._build_model(image, params['num_classes'], is_train=True, save_var=save_var)

        # FCN-32s
        upscore32 = nn.upscore_layer(model['score_fr'],      # output from last layer
                                     "upscore32",
                                     tf.shape(image),   # reshape to original input image size
                                     params['num_classes'],
                                     ksize=64, stride=32)
        old_shape = tf.shape(upscore32)
        new_shape = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']]
        prediction = tf.reshape(upscore32, new_shape)
        # num_pixels = tf.cast(new_shape[0], tf.int64)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, truth))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss

    def train_fcn16(self, params, image, truth,save_var=False):
        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        '''

        model = self._build_model(image, params['num_classes'], is_train=True, save_var=save_var)
        # upsample the last layer, train this, but don't save trained weights.
        upscore2_fr = nn.upscore_layer(model['score_fr'],
                                           "upscore2_fr",
                                           tf.shape(model['pool4']),
                                           params['num_classes'],
                                           ksize=4, stride=2)

        # Fuse fc8 *2, pool4, random to 0, train this, save trained weights
        in_features = model['pool4'].get_shape()[3].value
        score_pool4 = nn.conv_layer(model['pool4'], self.data_dict, name="score_pool4" ,shape=[1, 1, in_features, params['num_classes']], relu=False, dropout=False, var_dict=self.var_dict)

        # just simple adding.
        fuse_pool4 = tf.add(upscore2_fr, score_pool4)

        # Upsample fusion *16, the final upsampling, train this, but don't save trained weights.
        upscore16 = nn.upscore_layer(fuse_pool4,
                                         "upscore16",
                                         tf.shape(image),
                                         params['num_classes'],
                                         ksize=32, stride=16)

        old_shape = tf.shape(upscore16)
        new_shape = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']]
        prediction = tf.reshape(upscore16, new_shape)
        # num_pixels = tf.cast(new_shape[0], tf.int64)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, truth))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss

