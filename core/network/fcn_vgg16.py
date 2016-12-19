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
        # Load VGG16 pretrained weight
        data_dict = dt.load_vgg16_weight(data_path)
        self.data_dict = data_dict

        # used to save trained weights
        self.var_dict = {}

        # Init other necessary parameters
    def _build_model(self, image, num_classes, is_train=False, random_init_fc8=False, save_var=False):
        model = {}
        feed_dict = self.data_dict

        if save_var:
            var_dict = self.var_dict
        else:
            var_dict = None

        print('Save_var',save_var)

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

        '''
        For fconv6_*, we use feed_names to specify the keys of weight and bias to load from feed_dict.
        If feed_name = None, weight and bias are randomly initialized.
        '''
        if is_train:
            # Load from vgg16 during training
            feeds_name = {'fc6_1':'fc6_1', 'fc6_2':'fc6_2', 'fc6_3':'fc6_3'}

        else:
            # Load from trained fcn32s during inference
            feeds_name = {'fc6_1':'fc6_1', 'fc6_2':'fc6_2', 'fc6_3':'fc6_3'}


        # [7, 7, 512, 4096] Replace 7*7 conv kernel with 3 3*3 conv kernals
        model['fconv6_1'] = nn.fully_conv_layer(model['pool5'], feed_dict,
                                                feed_name=feeds_name['fc6_1'], name="fc6_1",
                                                shape=[3, 3, 512, 512],
                                                dropout=is_train, keep_prob=0.5,
                                                var_dict=var_dict)

        model['fconv6_2'] = nn.fully_conv_layer(model['fconv6_1'], feed_dict,
                                                feed_name=feeds_name['fc6_2'], name="fc6_2",
                                                shape=[3, 3, 512, 512],
                                                dropout=is_train, keep_prob=0.5,
                                                var_dict=var_dict)

        model['fconv6_3'] = nn.fully_conv_layer(model['fconv6_2'], feed_dict,
                                                feed_name=feeds_name['fc6_3'], name="fc6_3",
                                                shape=[3, 3, 512, 4096],
                                                dropout=is_train, keep_prob=0.5,
                                                var_dict=var_dict)

        model['fconv7'] = nn.fully_conv_layer(model['fconv6_3'], feed_dict,
                                                feed_name="fc7", name="fc7",
                                                shape=[1, 1, 4096, 4096],
                                                dropout=is_train, keep_prob=0.5,
                                                var_dict=var_dict)

        # model['fconv7'] = nn.fully_conv_layer(model['fconv6_3'], feed_dict, feed_name="fc7", name="fc7", vgg_f7_shape, dropout=is_train, keep_prob=0.5, var_dict=var_dict)

        if random_init_fc8:
            # Randomly init fc8
            feed_name_score_fr = None
        elif feed_dict.has_key('score_fr'):
            # If we are using fcn32s trained weight
            feed_name_score_fr = 'score_fr'
        else:
            # If we are using vgg16 trained weight
            feed_name_score_fr = 'fc8'

        model['score_fr'] = nn.score_layer(model['fconv7'], feed_dict,
                                           feed_name=feed_name_score_fr,
                                           name="score_fr",
                                           num_classes=num_classes,
                                           stddev=0.001, var_dict=var_dict)
        return model

    '''
    def save_weights(self, sess=None, npy_path=None):
        assert isinstance(sess, tf.Session)
        assert npy_path != None

        if sess == None or npy_path == None:
            print("No valid session or path! Saving file aborted!")
        else:
            data_dict = {}

            for (name, idx), var in self.var_dict.items():
                var_out = sess.run(var)
                if not data_dict.has_key(name):
                    data_dict[name] = {}
                data_dict[name][idx] = var_out
            np.save(npy_path, data_dict)

        print("trained weights saved: ", npy_path)
        return npy_path
    '''

    def inference(self, image, num_classes, random_init_fc8=False,option={'fcn32s':True, 'fcn16s':False, 'fcn8s':False}):
        # Image preprocess: RGB -> BGR
        red, green, blue = tf.split(3, 3, image)
        image = tf.concat(3, [blue, green, red])

        # Basic model
        model = self._build_model(image, num_classes, is_train=False, random_init_fc8=random_init_fc8)

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
            score_pool4 = nn.score_layer(model['pool4'],
                                         feed_dict=self.data_dict,
                                         feed_name='score_pool4',     # Use trained pool4 score weights
                                         name='score_pool4',
                                         num_classes=num_classes,
                                         stddev=0.001)

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
            score_pool3 = nn.score_layer(model['pool3'],
                                         feed_dict=self.data_dict,
                                         feed_name=None,     # Random initialize
                                         name='score_pool3',
                                         num_classes=num_classes,
                                         stddev=0.001)


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
    def train_fcn32(self, params, image, truth, random_init_fc8=True,save_var=False):

        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        '''

        # Important: When training, random_init_fc8=True. When inference, random_init_fc8=False
        model = self._build_model(image, params['num_classes'], is_train=True, random_init_fc8=random_init_fc8, save_var=save_var)

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

    def train_fcn16(self, params, image, truth, random_init_fc8=True,save_var=False):
        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        '''

        model = self._build_model(image, params['num_classes'], is_train=True, random_init_fc8=random_init_fc8, save_var=save_var)
        # upsample the last layer, train this, but don't save trained weights.
        upscore2_fr = nn.upscore_layer(model['score_fr'],
                                           "upscore2_fr",
                                           tf.shape(model['pool4']),
                                           params['num_classes'],
                                           ksize=4, stride=2)

        # Fuse fc8 *2, pool4, random to 0, train this, save trained weights
        score_pool4 = nn.score_layer(model['pool4'],
                                         feed_dict=self.data_dict,
                                         feed_name=None,     # Random initialize
                                         name='score_pool4',
                                         num_classes=params['num_classes'],
                                         stddev=0.001,
					 var_dict=self.var_dict)
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

