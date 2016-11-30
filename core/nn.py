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

# for testing 
import data_utils as dt


def test():
    data_dict = dt.load_vgg16_weight('data')
    name = 'fc7'
    print(data_dict[name][0].shape, len(data_dict[name][1]))
    #conv_layer()
    
test()

"""
 Dict keys:
['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
"""

def max_pool_layer(x, name, stride = 2):
    pool = tf.nn.max_pool(x, ksize = [1, stride, stride, 1], padding = 'SAME', name = name)
    return pool

def conv_layer(x, feed_dict, name, stride = 1):
    with tf.variable_scope(name) as scope:
        kernel = get_conv_kernel(feed_dict, name)
        conv = tf.nn.conv2d(x, kernel, strides = [1, stride, stride, 1], padding = 'SAME')
        bias = get_bias(name)
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias), name = scope.name)
        return conv_out


def fully_conv_layer(x, feed_dict, name, num_classes=None, relu=True, dropout=False, keep_prob=0.5):
    # QJ Not finished
    shape = [1, 1, 4096, 4096]
    with tf.variable_scope(name) as scope:
        kernel = get_conv_kernel(name)
        conv = tf.nn.conv2d(x, kernel, strides = [1, stride, stride, 1], padding = 'SAME')
        bias = get_bias(name)
        conv_out = tf.nn.bias_add(conv, bias)
        
        if relu:
            conv_out =  tf.nn.relu(conv_out)

        if dropout:
            conv_out = tf.nn.drop(conv_out, keep_prob)
            
        return conv_out


def score_layer(x, feed_dict, name, num_classes):
    # QJ Not finished    
    pass


def upscore_layer(x, feed_dict, name, ksize=4, stride=2):
    # WY
    pass


def get_conv_kernel(feed_dict, name):
    kernel = feed_dict[name][0]
    shape = kernel.shape
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))
   
    init = tf.constant_initializer(value=kernel,dtype=tf.float32)   
    var = tf.get_variable(name="kernel", initializer=init, shape=shape)
    
    '''
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                              name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    '''
    return var


def get_bias(feed_dict, name):
    bias = feed_dict[name][1]
    shape = bias.shape
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))

    init = tf.constant_initializer(value=bias, dtype=tf.float32)   
    var = tf.get_variable(name="bias", initializer=init, shape=shape)
    
    return var

def get_fconv_weight(feed_dict, name, shape, num_class=None):
    weight = feed_dict[name][0]
    weight = weight.reshape(shape)
    shape = weight.shape    
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))
    
    init = tf.constant_initializer(value=weight, dtype=tf.float32)   
    var = tf.get_variable(name="weight", initializer=init, shape=shape)
    # 
    return var

def get_deconv_filter(shape):
    # WY
    # Bilinear interpolation 
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
    return var

def get_bias_variable(name, shape, const = 0.1):
    initializer = tf.constant_initializer(const)
    tf.get_variable(name = name, shape = shape, initializer = initializer)
    pass


