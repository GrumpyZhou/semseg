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
import math
import data_utils as dt

DATA_DIR = 'data'

class InstanceSensitiveFCN8s:

    def __init__(self, data_path=None):
        # Load pretrained weight
        data_dict = dt.load_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}


    def _build_model(self, image, is_train=False, save_var=False):

        model = {}
        # During training, use pretrained vgg16 weight
        feed_dict = self.data_dict

        if save_var:
            # During training, weights are saved to var_dict
            var_dict = self.var_dict
        else:
            # During inference or validation, no need to save weights
            var_dict = None

        # Base channel: build fcn8s and produce feature maps with shape[H, W, 512]
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

        # Skip feature fusion
        model['score_fr'] = nn.conv_layer(model['conv7'], feed_dict, "score_fr_inst",
                                          shape=[1, 1, 4096, 512], relu=False,
                                          dropout=False, var_dict=var_dict)

        # Upsample: score_fr*2
        upscore_fr_2s = nn.upscore_layer(model['score_fr'], feed_dict, "upscore_fr_2s_inst",
                                       tf.shape(model['pool4']), num_class=512,
                                       ksize=4, stride=2, var_dict=var_dict)
        # Fuse upscore_fr_2s + score_pool4
        in_features = model['pool4'].get_shape()[3].value
        score_pool4 = nn.conv_layer(model['pool4'], feed_dict, "score_pool4_inst",
                                    shape=[1, 1, in_features, 512],
                                    relu=False, dropout=False, var_dict=var_dict)

        fuse_pool4 = tf.add(upscore_fr_2s, score_pool4)


        # Upsample fuse_pool4*2
        upscore_pool4_2s = nn.upscore_layer(fuse_pool4, feed_dict, "upscore_pool4_2s_inst",
                                            tf.shape(model['pool3']), num_class=512,
                                            ksize=4, stride=2, var_dict=var_dict)

        # Fuse upscore_pool4_2s + score_pool3
        in_features = model['pool3'].get_shape()[3].value
        score_pool3 = nn.conv_layer(model['pool3'], feed_dict, "score_pool3_inst",
                                    shape=[1, 1, in_features, 512],
                                    relu=False, dropout=False, var_dict=var_dict)

        model['score_out'] = tf.add(upscore_pool4_2s, score_pool3)


        # Instance assembling score
        model['inst_conv1'] = nn.conv_layer(model['score_out'], feed_dict, "inst_conv1",
                                            shape=[1, 1, 512, 512],relu=False,
                                            dropout=False, var_dict=var_dict)

        model['inst_score'] = nn.conv_layer(model['inst_conv1'], feed_dict, "inst_conv2",
                                            shape=[3, 3, 512, 9],relu=False,
                                            dropout=False, var_dict=var_dict)

        # Objectness score
        model['obj_conv1'] = nn.conv_layer(model['score_out'], feed_dict, "obj_conv1",
                                            shape=[3, 3, 512, 512],relu=False,
                                            dropout=False, var_dict=var_dict)

        model['obj_score'] = nn.conv_layer(model['obj_conv1'], feed_dict, "obj_conv2",
                                            shape=[1, 1, 512, 1],relu=False,
                                            dropout=False, var_dict=var_dict)

        """
        # Upsample to original size *8 # Or we have to do it by class
        model['upmask'] = nn.upscore_layer(score_out, feed_dict,
                                  "upmask", tf.shape(image), self.num_pred_class * max_instance,
                                  ksize=16, stride=8, var_dict=var_dict)
        """


        print('Instance-sensitive-fcn8s model is builded successfully!')
        print('Model: %s' % str(model.keys()))
        return model

    def inference(self, image, top_k=5, sz=21, stride=8, w=128, h=256):
        """
        Input: image
        Return: a stack of masks, shape = [h, w, num_classes],
                each slice represent instance masks belonging to a class
                value of each pixel is between [0,max_instance)
        """
        # Build model
        model = self._build_model(image, is_train=False)

        """Assemble instance proposals during inference by densely sliding to generate proposals """
        obj_score = model['obj_score']
        inst_score = model['inst_score']
        inst_shape = tf.shape(inst_score)

        ix = int((w - sz) / stride)
        iy = int((h - sz) / stride)
        proposal_map = []
        object_pos = []
        object_scores = []

        # Sliding over score map
        pos_x = 0
        for i in range(ix):
            pos_y = 0
            for j in range(iy):
                # record position
                object_pos.append((pos_x, pos_y))

                # calculate objectness score
                objectness = tf.slice(obj_score, [0, pos_x, pos_y, 0], [1, sz, sz, 1])
                score = tf.reshape(tf.reduce_mean(objectness), [1]) # reshape is necessary to perform for tf.concat
                object_scores.append(score)

                pos_y += stride + sz
                if ((pos_y + sz) >= h):
                    break

            pos_x += stride + sz
            if ((pos_x + sz) >= w):
                break

        # get top_k result, returns: [vals, indices]
        indice = tf.nn.top_k(tf.concat(0,object_scores), top_k)[1]
        position = tf.constant(object_pos, tf.int32) # for indexing within tf

        # generate top_k instance proposals
        for i in range(top_k):
            pos = position[indice[i]]
            sub_score = tf.slice(inst_score, [0, pos[0], pos[1], 0], [1, sz, sz, inst_shape[3]])
            instance = assemble(sz, sz, sub_score, k=3)

            # Upsample instance proposal to original size *8
            instance = nn.upscore_layer(instance, {}, "upinst_inf", 
                                        tf.shape([1, w*8, h*8, 1]), num_class=1,
                                        ksize=16, stride=8, var_dict=None)

            proposal_map.append(instance)

        print('Total proposals %d'%len(proposal_list))
        return proposal_map


    def train(self, image, gt_mask, gt_box, learning_rate=1e-6, num_box = 256, save_var=True):
        '''
        Input
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32
        gt_masks: stacked instance_masks, shape=[1, h, w, num_gt_class], tf.int32
        '''
        # Build basic model
        model = self._build_model(image, is_train=True, save_var=save_var)

        # Assemble instance proposals during training with given location and size of bounding boxs
        inst_score = model['inst_score']
        obj_score = model['obj_score']
        loss = 0

        for k in range(num_box):
            # box location and size
            x = gt_box[k][0][0]
            y = gt_box[k][0][1]
            w = gt_box[k][1][0]
            h = gt_box[k][1][1]

            obj_score_gt = gt_box[k][2][0]
            obj_id = gt_box[k][2][1]
            cond = tf.equal(1, obj_score_gt)

            # if it is a positive sample, score_gt is 1 else 0
            w_s =  tf.cast(tf.floor(w / 8), tf.int32)
            h_s =  tf.cast(tf.floor(h / 8), tf.int32)

            obj_score_gt = tf.cond(cond, lambda: tf.constant(1, tf.float32), lambda: tf.constant(0, tf.float32))
            x_s = tf.cast(tf.floor(x / 8), tf.int32)
            y_s = tf.cast(tf.floor(y / 8), tf.int32)
            obj_score_pred = tf.reduce_mean(tf.slice(obj_score, [0, x_s, y_s, 0], [1, w_s, h_s, 1]))
            #obj_score_pred = tf.to_int32(obj_score_pred)
            obj_loss = tf.abs(obj_score_gt - obj_score_pred)
            #obj_loss = tf.cast(obj_loss, tf.float32)

            inst_loss = tf.cond(cond, lambda: self.get_inst_loss(inst_score, gt_mask, x, y, w, h, obj_id), lambda: tf.constant(0, tf.float32))
            loss += obj_loss + inst_loss

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step, loss

    def get_inst_loss(self, inst_score, gt_mask, x, y, w, h, obj_id):

        # generate instance proposal
        inst_shape = tf.shape(inst_score)
        w_s =  tf.cast(tf.floor(w / 8), tf.int32)
        h_s =  tf.cast(tf.floor(h / 8), tf.int32)
        x_s = tf.cast(tf.floor(x / 8), tf.int32)
        y_s = tf.cast(tf.floor(y / 8), tf.int32)
        sub_score = tf.slice(inst_score, [0, x_s, y_s, 0], [1, w_s, h_s, inst_shape[3]])
        instance = self.assemble(w_s, h_s, sub_score)

        # Upsample instance proposal to original size *8
        instance = tf.image.resize_bilinear(instance, [w, h])
        #instance = tf.reshape(instance, [1, w, h, 1])


        # generate gt instance
        instance_gt_ = tf.slice(gt_mask, [0, x, y, 0], [1, w, h,1])
        indices = tf.where(tf.equal(instance_gt_, obj_id))
        sparse_val = tf.constant(1, dtype=tf.float32)
        instance_gt_shape = tf.to_int64([1, w, h, 1])
        instance_gt_shape = tf.pack(instance_gt_shape)
        instance_gt = tf.sparse_to_dense(indices, instance_gt_shape, sparse_val)

        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=instance_gt, logits=instance))


    def assemble(self, w, h, score, k=3):
        """
        Assemble  k*k parts across last dimension to make one instance proposal
        Return a tensor with the shape [1, w, h, 1] 
        """
        dx = tf.floor(w / k)
        dy = tf.floor(h / k)
        dx = tf.cast(dx, tf.int32)
        dy = tf.cast(dy, tf.int32)
        parts = []
        for i in range(k):
            for j in range(k):
                c = tf.constant(i*k+j, tf.int32)
                parts.append(tf.slice(score, [0, i*dx, j*dy, c], [1, dx, dy, 1]))

        # concat along x first, then concat along y
        concated_x = []
        for i in range(k*k):
            parts[i] = tf.reshape(parts[i], [dx,dy])
        for i in range(k):
            temp_x = tf.concat(1,[parts[i*3+0], parts[i*3+1], parts[i*3+2]])
            concated_x.append(temp_x)
        instance = tf.concat(0,concated_x)

        instance = tf.reshape(instance, [1, dx*k, dy*k, 1])
        return instance

