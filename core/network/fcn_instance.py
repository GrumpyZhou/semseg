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

class InstanceFCN8s:

    def __init__(self, data_path=None, pred_class={11:'person', 13:'car'}, gt_class={11:'person', 13:'car'}):
        # Define classes to be segmented to instance level e.g {11:'person', 13:'car'}
        self.gt_class = gt_class
        self.pred_class = pred_class
        self.num_pred_class = len(pred_class)
        self.num_gt_class = len(gt_class)


        # Load pretrained weight
        data_dict = dt.load_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}


    def _build_model(self, image, num_classes, max_instance, direct_slice, is_train=False, save_var=False, val_dict=None):

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

        # For testing, generate semantic by upsample fusion *8
        """fcn8s = nn.upscore_layer(score_out, feed_dict, "upscore8",
                                 tf.shape(image), num_classes,
                                 ksize=16, stride=8, var_dict=var_dict)
        
	model['fcn8s'] = tf.squeeze(fcn8s)"""
        sub_score = []
        shape = tf.shape(score_out)

        if direct_slice:
            # Select directly the designated classes e.g car and person
            for id in sorted(self.target_class.keys()):
                print('slicing %d'%id)
                sub_score.append(tf.slice(score_out, [0, 0, 0, id], [shape[0], shape[1], shape[2], 1]))
        else:
            # Perform prediction and generate corresponding masks of designated classes e.g car and person
            pred_out = tf.argmax(score_out, dimension=3)
            pred_out = tf.cast(pred_out, tf.float32)
            pred_out = tf.reshape(pred_out,[shape[1],shape[2]])

            for id in sorted(self.target_class.keys()):
                where = tf.equal(pred_out, id)
                indices = tf.where(where)
                val = tf.ones((tf.shape(indices)[0],),tf.float32)
                mask = tf.sparse_to_dense(indices,[128, 256],val)
                mask = tf.reshape(mask, [shape[0], shape[1], shape[2], 1])
                sub_score.append(mask)

        model['semantic_mask'] = tf.concat(3, sub_score)

        # Convolve semantic_mask to several stacks of instance masks, each having shape [1, h, w, max_instance]
        model['instance_mask'] = nn.mask_layer(model['semantic_mask'], feed_dict, "conv_depth_mask",
                                               shape=[3, 3, self.num_pred_class, max_instance],
                                               relu=False, dropout=False, var_dict=var_dict)

        # Upsample to original size *8 # Or we have to do it by class
        model['upmask'] = nn.upscore_layer(model['instance_mask'], feed_dict,
                                  "upmask", tf.shape(image), self.num_pred_class * max_instance,
                                  ksize=16, stride=8, var_dict=var_dict)

        print('InstanceFCN8s model is builded successfully!')
        print('Model: %s' % str(model.keys()))
        return model

    def train(self, params, image, gt_masks, direct_slice=True, save_var=True):
        '''
        Input
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32
        gt_masks: stacked instance_masks, shape=[1, h, w, num_gt_class], tf.int32
        '''
        # Build model
        model = self._build_model(image, params['num_classes'], params['max_instance'], direct_slice=direct_slice, is_train=True, save_var=save_var)
        pred_masks = model['upmask']
        # Split stack by semantic class
        pred_mask_list = tf.split(3, self.num_pred_class, pred_masks)
        gt_mask_list = tf.split(3, self.num_gt_class, gt_masks)

        # Loss: softmax + cross entropy
        loss = 0
        total_pixel = 1024 * 2048

        """Training only with car """
        pred = pred_mask_list[0]
        gt = gt_mask_list[1]
        for j in range(params['max_instance']):
            shape = tf.shape(pred)
            pred_j = tf.slice(pred, [0, 0, 0, j], [shape[0], shape[1], shape[2], 1])
            (gt_j, inst_pixel) = self.get_gtmask_tuple(gt, j+1)
            weight = tf.cond(tf.equal(inst_pixel, 0), lambda: tf.constant(1, dtype=tf.float64) , lambda: (total_pixel - inst_pixel) / inst_pixel )
            weight = tf.cast(weight, tf.float32)
            weight = tf.cast(weight, tf.float32)
            pred_j = tf.cast(pred_j, tf.float32)
            gt_j = tf.cast(gt_j, tf.float32)
            # loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_j, gt_j)) 
            loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_j, gt_j, weight))

        """   Multiclasses
        for i in range(self.num_pred_class):
            pred = pred_mask_list[i]
            gt = gt_mask_list[i]
            for j in range(params['max_instance']):
                shape = tf.shape(pred)
                pred_j = tf.slice(pred, [0, 0, 0, j], [shape[0], shape[1], shape[2], 1])
                (gt_j, inst_pixel) = self.get_gtmask_tuple(gt, j+1)
                weight = tf.cond(tf.equal(inst_pixel, 0), lambda: tf.constant(1, dtype=tf.float64) , lambda: (total_pixel - inst_pixel) / inst_pixel )
                weight = tf.cast(weight, tf.float32)
                weight = tf.cast(weight, tf.float32)
                pred_j = tf.cast(pred_j, tf.float32)
                gt_j = tf.cast(gt_j, tf.float32)
                # loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_j, gt_j)) 
                loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_j, gt_j, weight))
        """
        
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        # For testing 
        #fcn8s = tf.argmax(model['fcn8s'], dimension=2)
        return train_step, loss

    def inference(self, params, image, direct_slice=True):
        """
        Input: image
        Return: a stack of masks, shape = [h, w, num_classes],
                each slice represent instance masks belonging to a class
                value of each pixel is between [0,max_instance)
        """
        # Build model
        model = self._build_model(image, params['num_classes'], params['max_instance'], direct_slice=direct_slice, is_train=False)
        pred_masks = model['upmask']

        # Split stack by semantic class
        pred_mask_list = tf.split(3, self.num_pred_class, pred_masks)
        instance_masks = []
        for i in range(self.num_pred_class):
            pred = tf.argmax(pred_mask_list[i], dimension=3)
            instance_masks.append(tf.squeeze(pred))
        return instance_masks

    def get_gtmask_tuple(self, gt_sparse, inst_id):
        '''
        gt_sparse: [1, H, W, 1] sparse gt mask
        inst_id: 1-30 (MUST not start from 0!)
        Return: tuple (0/1 mask, #1s)
        '''
        gt_shape = tf.shape(gt_sparse)
        gt_sparse = tf.reshape(gt_sparse, [gt_shape[1], gt_shape[2], 1])
        where = tf.equal(gt_sparse, inst_id)
        indices = tf.where(where)
        val = tf.ones((tf.shape(indices)[0],),tf.float32)
        mask = tf.sparse_to_dense(indices,[1024, 2048,1],val)
        mask = tf.reshape(mask, [1, 1024, 2048, 1])

        return (mask, tf.shape(indices)[0])
