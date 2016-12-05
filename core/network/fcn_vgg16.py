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
    def train(self, params, batch, label, num_pixels, sparse_values, add_bias):
        '''
        Note Dtype:
        batch: reshaped image value, shape=[1, Height, Width, 3], tf.float32, numpy ndarray
        label: reshaped image label, shape=[Height*Width], tf.int32, numpy ndarray
        sparse_matrix: sparse-diagonal! shape=Height*Width, tf.float32,  scipy spare-diag
        add_bias: {0,1} vector, shape = shape=[Height*Width], tf.int32, numpy ndarray
        '''
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
        # num_pixels = tf.cast(new_shape[0], tf.int64)

        # truth has uint8 type, has to be casted to tf.int32 to cal loss
        # truth = tf.reshape(label, [new_shape[0]])
        # truth_ = tf.cast(truth, tf.int32)

        # apply sparse matrix multiplication, only available for float32 dtype
        # sparse_matrix_ = tf.cast(sparse_matrix, tf.float32)
        # pred_mul = tf.matmul(sparse_matrix_, prediction, a_is_sparse=True)
        single_indices = np.arange(num_pixels)
        single_indices_ = tf.cast(single_indices, tf.int64)
        single_indices_f = tf.reshape(single_indices_, [num_pixels, 1])

        # single_indices_ = tf.reshape(single_indices, [new_shape[0], 1])
        sparse_indices = tf.concat(1, [single_indices_f, single_indices_f])
        spare_diag_matrix = tf.SparseTensor(sparse_indices, sparse_values, [num_pixels, num_pixels])
        pred_mul = tf.sparse_tensor_dense_matmul(spare_diag_matrix, prediction)

        # slice the last column and add it to the add_bias_reshaped
        add_bias_ = tf.cast(add_bias, tf.float32)
        last_col = tf.add(tf.slice(pred_mul, [0,22], [-1,1]), add_bias_)
        # slice the first 21 columns
        matrix_left = tf.slice(pred_mul, [0,0], [-1, params['num_classes']-1])
        # concatenate the first 21 columns and the last column
        last_col_ = tf.reshape(last_col, [num_pixels, 1])
        pred_final = tf.concat(1, [matrix_left, last_col_])


        # find all indices where the pixel label is 255,
        # this must be excluded from calculating cross-entropy
        # truth_1 = tf.contrib.util.make_tensor_proto(truth_)
        # truth_array = tf.contrib.util.make_ndarray(truth_.eval(sess))
        # truth_array = truth_.eval(sess)
        # ii = tf.where(truth_ == 255)   # find all indices where element value is 255
        # dim = old_shape[1] * old_shape[2]
        # xx = tf.ones( [dim] )
        # np.put(xx, ii, [0])
        # yy = tf.diag(xx)
        # truth_1 = tf.matmul(yy, prediction)

        # # vector to be concatenated to zero matrix
        # xx_1 = tf.zeros([old_shape[1]*old_shape[2]])
        # np.put(xx_1, ii, [1])

        # # create zero matrix and concatenate with a vector
        # new_shape_ = [old_shape[0]*old_shape[1]*old_shape[2], params['num_classes']-1]
        # yy1 = tf.zeros(new_shape_, dtype=tf.float32)
        # yy1_t = tf.transpose(yy1)
        # yy1_con = tf.concat(0,xx_1)
        # yy1_final = tf.tranpose(yy1_con)

        # # create final prediction
        # prediction_ = tf.add(truth_1, yy1_final)

        # truth_array_ =  np.delete(truth_array, ii)  # delete all elements equal to 255
        # # # the same preprocessing for predictions
        # prediction_array = tf.contrib.util.make_ndarray(prediction)
        # prediction_array_ = np.delete(prediction_array, ii, 0)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred_final, label))
        train_step = tf.train.AdamOptimizer(params['rate']).minimize(loss)

        return train_step, loss
