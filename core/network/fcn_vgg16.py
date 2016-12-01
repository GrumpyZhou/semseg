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
        """Dict keys:['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3','conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']"""
        # Load VGG16 pretrained weight
        data_dict = dt.load_vgg16_weight(data_dir)
        self.data_dict = data_dict

        # Init other necessary parameters
    def _build_model(self, feed_dict, image, is_train=True, random_init_fc8=False):
        model = {}

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


        model['fconv6'] = nn.fully_conv_layer(model['pool5'], feed_dict, "fc6", [7, 7, 512, 4096], dropout=is_train, keep_prob=0.5])
        model['fconv7'] = nn.fully_conv_layer(model['fconv6'], feed_dict, "fc7", [1, 1, 4096, 4096], dropout=is_train, keep_prob=0.5])

        num_classes = 1000
        model['score_fr'] = nn.score_layer(model['fconv7'], "score_fr", [1, 1, 4096, num_classes], random=random_init_fc8, feed_dict=feed_dict)
        return model

    def inference(self, image, num_classes):
        # Not finished QJ

        # Image preprocess: RGB -> BGR
        red, green, blue = tf.split(3, 3, image)
        image = tf.concat(3, [blue, green, red])

        # Network structure -- VGG16
        # Pretrained weight on imageNet
        feed_dict = self.data_dict

        # Basic model
        model = self._build_model( image, is_train=False)

        # FCN-32s
        # Inconsistent with definition
        # upscore32 = nn.upscore_layer(model['score_fr'],
        #                             shape=tf.shape(model['pool4']),
        #                             num_classes=num_classes,
        #                             name="upscore32",
        #                             ksize=4, stride=2)
        # Use this one:
        upscore32 = nn.upscore_layer(model['score_fr'],       # output from last layer
                                    name="upscore32",
                                    shape=tf.shape(image),    # original size of input image
                                    # num_classes=num_classes, already encoded in output from last layer
                                    ksize=4, stride=2)
        # Another version of upsampling. Prefer to use upscore_layer() first
        # upscore32 = _upscore_layer(model['score_fr'],
        #                            shape=tf.shape(image),
        #                            num_class=num_classes,
        #                            name="upscore32",
        #                            ksize=4, stride=2)


        # FCN-16s
        shape_p4 = model['pool4'].get_shape().as_list()
        shape_p4.append(num_classes)
        score_pool4 = nn.score_layer(model['pool4'], "score_pool4", shape_p4, random=random_init_fc8, feed_dict=feed_dict)
        fuse_pool4 = tf.add(upscore32, score_pool4)

        # Inconsistent with definition
        # upscore16 = nn.upscore_layer(fuse_pool4,
        #                              shape=tf.shape(image),
        #                              num_classes=num_classes,
        #                              name="upscore16",
        #                              ksize=32, stride=16)
        # Use this one:
        upscore16 = nn.upscore_layer(fuse_pool4,
                            name="upscore16",
                            shape=tf.shape(image),
                            ksize=32, stride=16)
        # upscore16 = _upscore_layer(fuse_pool4,
        #                            shape=tf.shape(image),
        #                            num_class=num_classes,
        #                            name="upscore16",
        #                            ksize=32, stride=16)

        ''' FCN-8s
        shape_p3 = model['pool3'].get_shape().as_list()
        shape_p3.append(num_classes)
        score_pool3 = nn.score_layer(model['pool3'], "score_pool3", shape_p3, random=random_init_fc8, feed_dict=feed_dict)
        fuse_pool3 = tf.add(upscore16, score_pool3)
        upscore8 = nn.upscore_layer(fuse_pool4,
                                     shape=tf.shape(image),
                                     num_classes=num_classes,
                                     name="upscore16",
                                     ksize=32, stride=16)
        '''
        # Prediction
        pred_32s = tf.argmax(model['score_fr'], dimension=3)
        pred_16s = tf.argmax(upscore32, dimension=3)

        pass

    def train(self, total_loss, learning_rate ):
        # To be implemented Later
        # Mini-batch
        # Minimize loss
        # Add necessary params to summary
        # Return train_op
        pass
