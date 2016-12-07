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

    def __init__(self, data_dir=None, data_name=None):
        # Load VGG16 pretrained weight
        data_dict = dt.load_vgg16_weight(data_dir, data_name)
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
        
        # Vgg use the filter because the input image is fixed size: 224 * 224, but it might affect our training because our image size vary
        vgg_f6_shape = [7, 7, 512, 4096]
        vgg_f7_shape = [1, 1, 4096, 4096] # we can stick to this one
        model['fconv6'] = nn.fully_conv_layer(model['pool5'], feed_dict, "fc6", vgg_f6_shape, dropout=is_train, keep_prob=0.5, var_dict=var_dict)
        model['fconv7'] = nn.fully_conv_layer(model['fconv6'], feed_dict, "fc7", vgg_f7_shape, dropout=is_train, keep_prob=0.5, var_dict=var_dict)

        model['score_fr'] = nn.score_layer(model['fconv7'], "score_fr", num_classes, random=random_init_fc8, feed_dict=feed_dict, var_dict=var_dict)
        
        return model

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

    # train model with an accuracy of 32-stride
    def train_fcn32(self, params, image, truth, diag_indices, diag_values, add_bias, save_var=False):

        '''
        Note Dtype:
        image: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        truth: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        diag_indices: required by tf.SparseTensor(), shape=[Height*Width, 2], tf.int64, numpy ndarray
        diag_values: required by tf.SparseTensor(), shape=[Height*Width], tf.float32, numpy ndarray
        add_bias: {0,1} vector, shape = shape=[Height*Width], tf.float32, numpy ndarray
        '''

        # Important: When training, random_init_fc8=True. When inference, random_init_fc8=False
        model = self._build_model(image, params['num_classes'], is_train=True, random_init_fc8=True, save_var=save_var)

        # FCN-32s
        upscore32 = nn.upscore_layer(model['score_fr'],      # output from last layer
                                     "upscore32",
                                     tf.shape(image),   # reshape to original input image size
                                     params['num_classes'],
                                     ksize=64, stride=32)
        old_shape = tf.shape(upscore32)
        new_shape = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']]
        prediction = tf.reshape(upscore32, new_shape)
        num_pixels = tf.cast(new_shape[0], tf.int64)

        spare_diag_matrix = tf.SparseTensor(diag_indices, diag_values, [num_pixels, num_pixels])
        pred_mul = tf.sparse_tensor_dense_matmul(spare_diag_matrix, prediction)

        add_bias_ = tf.reshape(add_bias, [new_shape[0], 1]) # convert to column vector
        last_col = tf.add(tf.slice(pred_mul, [0, params['num_classes']-1], [-1,1]), add_bias_)
        # slice the first 21 columns
        matrix_left = tf.slice(pred_mul, [0,0], [-1, params['num_classes']-1])
        # concatenate the first 21 columns and the last column
        # last_col_ = tf.reshape(last_col, [num_pixels, 1])
        pred_final = tf.concat(1, [matrix_left, last_col])

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred_final, truth))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss

    # use trained fcn32 model to inference
    def inference_fcn32(self, image, num_classes, random_init_fc8=False):
        '''
        Use trained fcn32 weights to do inference
        '''
        # Image preprocess: RGB -> BGR
        red, green, blue = tf.split(3, 3, image)
        image = tf.concat(3, [blue, green, red])

        # Basic model
        model = self._build_model(image, num_classes, is_train=False, random_init_fc8=False)

        # FCN-32s, return image size of [1, Height, Width, num_classes]
        upscore32 = nn.upscore_layer(model['score_fr'],      # output from last layer
                                     "upscore32",
                                     tf.shape(image),   # reshape to original input image size
                                     num_classes,
                                     ksize=64, stride=32)

        # Prediction
        pred32s = tf.argmax(upscore32, dimension=3)

        return pred32s

