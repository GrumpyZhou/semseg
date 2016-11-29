'''Compact interfaces lib for a neural network including:
-- Interfaces to define a nn layer e.g conv, pooling, relu, fcn, dropout etc
-- Interfaces for variable initialization
-- Interfaces for network data post-processing e.g logging, visualizing and so on
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

def test():
    print('nn')


def max_pool_layer(x, stride = 2, name = None):
    pool = tf.nn.max_pool(x, ksize = [
        1,
        stride,
        stride,
        1], trides = [
        1,
        stride,
        stride,
        1], padding = 'SAME', name = name)
    return pool


def conv_layer(x, feed_dict, name, stride = 1):
    with tf.variable_scope(name) as scope:
        kernel = get_conv_kernel(feed_dict, name)
        conv = tf.nn.conv2d(x, kernel, strides = [
            1,
            stride,
            stride,
            1], padding = 'SAME')
        bias = get_bias(name)
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias), name = scope.name)
        return conv_out


def fully_conv_layer(x, feed_dict, name, num_classes = None, relu = True):
    shape = [
        1,
        1,
        4096,
        4096]
    with tf.variable_scope(name) as scope:
        kernel = get_conv_kernel(name)
        conv = tf.nn.conv2d(x, kernel, strides = [
            1,
            stride,
            stride,
            1], padding = 'SAME')
        bias = get_bias(name)
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias), name = scope.name)
        return conv_out


def score_layer(x, feed_dict, name):
    pass


def deconv_layer(x, feed_dict, name):
    pass


def get_conv_kernel(feed_dict, key):
    pass


def get_bias(feed_dict, key):
    pass


def get_weight_variable_with_decay(name, shape, stddev = 0.1, wd = None):
    '''Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    '''
    initializer = tf.truncated_normal_initializer(stddev = stddev)
    var = tf.get_variable(name = name, shape = shape, initializer = initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection('losses', weight_decay)


def get_bias_variable(name, shape, const = 0.1):
    initializer = tf.constant_initializer(const)
    tf.get_variable(name = name, shape = shape, initializer = initializer)


