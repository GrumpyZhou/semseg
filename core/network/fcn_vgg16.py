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

    def __init__(self, data_dir=None):
        # Load VGG16 pretrained weight
        data_dict = dt.load_vgg16_weight(data_dir)
        self.data_dict = data_dict

        # Init other necessary parameters
    def _build_model(self, image, num_classes, is_train=False, random_init_fc8=False):
        model = {}
        feed_dict = self.data_dict

        model['conv1_1'] = nn.conv_layer(image, feed_dict, "conv1_1")
        model['conv1_2'] = nn.conv_layer(model['conv1_1'], feed_dict, "conv1_2")
        model['pool1'] = nn.max_pool_layer(model['conv1_2'], "pool1")

        model['conv2_1'] = nn.conv_layer(model['pool1'], feed_dict, "conv2_1")
        model['conv2_2'] = nn.conv_layer(model['conv2_1'], feed_dict, "conv2_2")
        model['pool2'] = nn.max_pool_layer(model['conv2_2'], "pool2")

        model['conv3_1'] = nn.conv_layer(model['pool2'], feed_dict, "conv3_1")
        model['conv3_2'] = nn.conv_layer(model['conv3_1'], feed_dict, "conv3_2")
        model['conv3_3'] = nn.conv_layer(model['conv3_2'], feed_dict, "conv3_3")
        model['pool3'] = nn.max_pool_layer(model['conv3_3'], "pool3")

        model['conv4_1'] = nn.conv_layer(model['pool3'], feed_dict, "conv4_1")
        model['conv4_2'] = nn.conv_layer(model['conv4_1'], feed_dict, "conv4_2")
        model['conv4_3'] = nn.conv_layer(model['conv4_2'], feed_dict, "conv4_3")
        model['pool4'] = nn.max_pool_layer(model['conv4_3'], "pool4")


        model['conv5_1'] = nn.conv_layer(model['pool4'], feed_dict, "conv5_1")
        model['conv5_2'] = nn.conv_layer(model['conv5_1'], feed_dict, "conv5_2")
        model['conv5_3'] = nn.conv_layer(model['conv5_2'], feed_dict, "conv5_3")
        model['pool5'] = nn.max_pool_layer(model['conv5_3'], "pool5")


        model['fconv6'] = nn.fully_conv_layer(model['pool5'], feed_dict, "fc6", [7, 7, 512, 4096], dropout=is_train, keep_prob=0.5)
        model['fconv7'] = nn.fully_conv_layer(model['fconv6'], feed_dict, "fc7", [1, 1, 4096, 4096], dropout=is_train, keep_prob=0.5)

        model['score_fr'] = nn.score_layer(model['fconv7'], "score_fr", num_classes, random=random_init_fc8, feed_dict=feed_dict)
        return model

    def inference(self, image, num_classes, random_init_fc8=False):
        # Image preprocess: RGB -> BGR
        red, green, blue = tf.split(3, 3, image)
        image = tf.concat(3, [blue, green, red])

        # Network structure -- VGG16
        # Pretrained weight on imageNet

        # Basic model
        model = self._build_model(image, num_classes, is_train=False, random_init_fc8=False)

        # FCN-32s
        upscore32 = nn.upscore_layer(model['score_fr'],      # output from last layer
                                     "upscore32",
                                     tf.shape(image),   # reshape to original input image size
                                     num_classes,
                                     ksize=64, stride=32)

        # FCN-16s
        upscore2_fr = nn.upscore_layer(model['score_fr'],       # output from last layer
                                       "upscore2_fr",
                                       tf.shape(model['pool4']),   # reshape to output of pool4
                                       num_classes,
                                       ksize=4, stride=2)

        # Fuse fc8 *2, pool4
        score_pool4 = nn.score_layer(model['pool4'],
                                     "score_pool4",
                                     num_classes,
                                     random=True,
                                     stddev=0.001,
                                     feed_dict=self.data_dict)
        fuse_pool4 = tf.add(upscore2_fr, score_pool4)

        # Upsample fusion *16
        upscore16 = nn.upscore_layer(fuse_pool4,
                                     "upscore16",
                                     tf.shape(image),   # reshape to original input image size
                                     num_classes,
                                     ksize=32, stride=16)

        # FCN-8s
        # Upsample fc8 *4
        upscore4_fr = nn.upscore_layer(fuse_pool4,    # output from last layer
                                       "upscore4_fr",
                                       tf.shape(model['pool3']),   # reshape to output of pool3
                                       num_classes,
                                       ksize=4, stride=2)
        # Upsample pool4 *2
        # upscore2_p4 = nn.upscore_layer(score_pool4,
        #                                "upscore2_p4",
        #                                tf.shape(image),
        #                                num_classes,
        #                                ksize=4, stride=2)

        # Fuse fc8 *4, pool4 *2, pool3
        score_pool3 = nn.score_layer(model['pool3'],
                                     "score_pool3",
                                     num_classes,
                                     random=True,
                                     stddev=0.001,
                                     feed_dict=self.data_dict)

        fuse_pool3 = tf.add(score_pool3, upscore4_fr)

        # # Upsample fusion *8
        upscore8 = nn.upscore_layer(fuse_pool3,
                                    "upscore8",
                                    tf.shape(image),    # reshape to original input image size
                                    num_classes,
                                    ksize=16, stride=8)


        # Prediction
        pred32s = tf.argmax(upscore32, dimension=3)
        pred16s = tf.argmax(upscore16, dimension=3)
        pred8s = tf.argmax(upscore8, dimension=3)


        return pred32s, pred16s, pred8s

    # def train(self, total_loss, learning_rate ):
    def train(self, params, batch, label):
        # To be implemented Later
        # Mini-batch
        # Minimize loss
        # Add necessary params to summary
        # Return train_op

        # Build the base model
        # batch_ = tf.reshape(batch, [1, tf.shape(batch)[2], tf.shape(batch)[3], 3])
        # print('shape of input: ', tf.shape(batch)[0], tf.shape(batch)[1], tf.shape(batch)[2], tf.shape(batch)[3])
        model = self._build_model(batch, params['num_classes'], is_train=True, random_init_fc8=True)

        # FCN-32s
        upscore32 = nn.upscore_layer(model['score_fr'],      # output from last layer
                                     "upscore32",
                                     tf.shape(batch),   # reshape to original input image size
                                     params['num_classes'],
                                     ksize=64, stride=32)
        old_shape = tf.shape(upscore32)
        new_shape = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']]
        prediction = tf.reshape(upscore32, new_shape)

        truth = tf.reshape(label, [new_shape[0]])
        # truth has uint8 type, has to be casted to tf.int32 to cal loss
        truth_ = tf.cast(truth, tf.int32)

        # find all indices where the pixel label is 255,
        # this must be excluded from calculating cross-entropy
        # truth_1 = tf.contrib.util.make_tensor_proto(truth_)
        '''
        truth_array = tf.contrib.util.make_ndarray(truth_)
        ii = tf.where(truth_array == 255)   # find all indices where element value is 255
        truth_array_ =  np.delete(truth_array, ii)  # delete all elements equal to 255
        # # the same preprocessing for predictions
        prediction_array = tf.contrib.util.make_ndarray(prediction)
        prediction_array_ = np.delete(prediction_array, ii, 0)
        '''

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, truth_))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss
