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

class InstanceSegNet:

    def __init__(self, data_path=None):
        # Load pretrained weight
        data_dict = dt.load_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}

    def _build_model(self, image, num_classes, is_train=False, scale_min='fcn16s', save_var=False, val_dict=None):
        
        model = {}
        if val_dict is None:
            # Not during validation, use pretrained weight
            feed_dict = self.data_dict
        else:
            # Duing validation, use the currently trained weight
            feed_dict = val_dict
            
        if save_var:
            # During training, weights are saved to var_dict
            var_dict = self.var_dict
        else:
            # During inference or validation, no need to save weights
            var_dict = None


        # Step1: build fcn8s and score_out which has shape[H, W, Classes] 
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

        model['conv6_1'] = nn.conv_layer(model['pool5'], feed_dict, "conv6_1", 
                                         shape=[3, 3, 512, 512], dropout=is_train, 
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv6_2'] = nn.conv_layer(model['conv6_1'], feed_dict, "conv6_2", 
                                         shape=[3, 3, 512, 512], dropout=is_train, 
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv6_3'] = nn.conv_layer(model['conv6_2'], feed_dict, "conv6_3", 
                                         shape=[3, 3, 512, 4096], dropout=is_train, 
                                         keep_prob=0.5, var_dict=var_dict)

        model['conv7'] = nn.conv_layer(model['conv6_3'], feed_dict, "conv7", 
                                       shape=[1, 1, 4096, 4096], dropout=is_train, 
                                       keep_prob=0.5, var_dict=var_dict)

        model['score_fr'] = nn.conv_layer(model['conv7'], feed_dict, "score_fr", 
                                          shape=[1, 1, 4096, num_classes], relu=False, 
                                          dropout=False, var_dict=var_dict)
        
        # Upsample: score_fr*2
        upscore_fr_2s = nn.upscore_layer(model['score_fr'], feed_dict, "upscore_fr_2s",
                                       tf.shape(model['pool4']), num_classes,
                                       ksize=4, stride=2, var_dict=var_dict)
        # Fuse upscore_fr_2s + score_pool4
        in_features = model['pool4'].get_shape()[3].value
        score_pool4 = nn.conv_layer(model['pool4'], feed_dict, "score_pool4", 
                                    shape=[1, 1, in_features, num_classes], 
                                    relu=False, dropout=False, var_dict=var_dict)
        
        fuse_pool4 = tf.add(upscore_fr_2s, score_pool4)

                  
        # Upsample fuse_pool4*2
        upscore_pool4_2s = nn.upscore_layer(fuse_pool4, feed_dict, "upscore_pool4_2s",
                                            tf.shape(model['pool3']), num_classes,
                                            ksize=4, stride=2, var_dict=var_dict)

        # Fuse  upscore_pool4_2s + score_pool3            
        in_features = model['pool3'].get_shape()[3].value
        score_pool3 = nn.conv_layer(model['pool3'], self.data_dict, "score_pool3", 
                                    shape=[1, 1, in_features, num_classes], 
                                    relu=False, dropout=False, var_dict=var_dict)

        score_out = tf.add(upscore_pool4_2s, score_pool3)

        # Conv score_out to masks which has shape [H, W, Num_Masks]
        num_masks = 40
        masks = nn.mask_layer(score_out, feed_dict, "masks", 
                                          shape=[3, 3, num_classes, num_masks], relu=False, 
                                          dropout=False, var_dict=var_dict)



     
        #self.var_dict = var_dict
        print('Model with scale %s is builded successfully!' % scale_min)
        print('Model: %s' % str(model.keys()))
        return model

    def inference(self, image, num_classes, scale_min='fcn16s', option={'fcn32s':False, 'fcn16s':True, 'fcn8s':False}):
        # Build model
        model = self._build_model(image, num_classes, is_train=False, scale_min=scale_min)
        
        # Keep using dictionary incase we want to compare results between different scales
        predict = {}
        for scale in option.keys():
            if option[scale]:
                predict[scale] = tf.argmax(model[scale], dimension=3)

        return predict

    def train(self, params, image, truth, scale_min='fcn16s', save_var=True):
        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        '''
        # Build model
        model = self._build_model(image, params['num_classes'], is_train=True, scale_min=scale_min, save_var=save_var) 
        upscored = model[scale_min]
        old_shape = tf.shape(upscored)
        new_shape = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']]
        prediction = tf.reshape(upscored, new_shape)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, truth))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss

    
