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
from math import ceil


def max_pool_layer(x, name, stride=2):
    pool = tf.nn.max_pool(x, ksize=[1, stride, stride, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME', name=name)
    return pool

def conv_layer(x, feed_dict, name, stride=1, shape=None, relu=True, dropout=False, keep_prob=0.5, var_dict=None):

    with tf.variable_scope(name) as scope:
        print('Layer name: %s' % name)  
        kernel = get_conv_kernel(feed_dict, name, shape)
        bias = get_bias(feed_dict, name, shape)

        conv = tf.nn.conv2d(x, kernel,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        conv_out = tf.nn.bias_add(conv, bias) 
            
        if relu:
            conv_out =  tf.nn.relu(conv_out)
        if dropout:
            conv_out = tf.nn.dropout(conv_out, keep_prob)

    if var_dict is not None:
        var_dict[name] = (kernel, bias)

    return conv_out

def atrous_conv_layer(x, feed_dict, name, rate=2, shape=None, relu=True, var_dict=None):

    with tf.variable_scope(name) as scope:
        print('Layer name: %s' % name)  
        kernel = get_conv_kernel(feed_dict, name, shape)
        bias = get_bias(feed_dict, name, shape)
 
        conv = tf.nn.atrous_conv2d(x, kernel, rate=rate, padding='SAME')
        conv_out = tf.nn.bias_add(conv, bias) 
            
        if relu:
            conv_out =  tf.nn.relu(conv_out)

    if var_dict is not None:
        var_dict[name] = (kernel, bias)

    return conv_out

# Use existing code, still don't understand. Prefer to use upscore_layer() first.
def upscore_layer(x, feed_dict, name, shape, num_class, ksize=4, stride=2, reuse_scope=False, var_dict=None):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if not reuse_scope:
            print('Layer name: %s' % name) 
        else:
            tf.get_variable_scope().reuse_variables() 
            
        in_features = x.get_shape()[3].value
        if shape is None:
            # Compute shape out of x
            in_shape = tf.shape(x)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_class]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_class]
        output_shape = tf.pack(new_shape)
        f_shape = [ksize, ksize, num_class, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        kernel = get_deconv_kernel(feed_dict, name, f_shape, reuse_scope)
        deconv = tf.nn.conv2d_transpose(x, kernel, output_shape,
                                        strides=strides, padding='SAME')
    if var_dict is not None:
        var_dict[name] = (kernel)

    return deconv

def get_conv_kernel(feed_dict, feed_name, shape):
    if not feed_dict.has_key(feed_name):
        print("No matched kernel %s, randomly initialize the kernel with shape: %s " % (feed_name, str(shape)))
        init = tf.constant_initializer(value=0, dtype=tf.float32)
	#init = tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32)
    else:
        kernel = feed_dict[feed_name][0]
        shape = kernel.shape
        print('Load kernel with shape: %s' % str(shape))
        init = tf.constant_initializer(value=kernel,dtype=tf.float32)
    var = tf.get_variable(name="kernel", initializer=init, shape=shape)
    return var

def get_bias(feed_dict, feed_name, shape):
    if not feed_dict.has_key(feed_name):
        shape = [shape[3]]        
        print("No matched bias %s, randomly initialize the bias with shape: %s " % (feed_name, str(shape)))
        init = tf.constant_initializer(0.1, dtype=tf.float32)
    else:
        bias = feed_dict[feed_name][1]
        shape = bias.shape
        print('Load bias with shape: %s' % str(shape))
        init = tf.constant_initializer(value=bias, dtype=tf.float32)        
        
    var = tf.get_variable(name="bias", initializer=init, shape=shape)
    return var
    
def get_deconv_kernel(feed_dict, feed_name, f_shape, reuse_scope=False):
    if not feed_dict.has_key(feed_name):
        if not reuse_scope:
            print("No matched deconv_kernel %s, use bilinear interpolation " % feed_name)
            
        # Bilinear interpolation
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        kernel = np.zeros(f_shape)
        for i in range(f_shape[2]):
            kernel[:, :, i, i] = bilinear
    else:
        kernel = feed_dict[feed_name]
        print('Load deconv_kernel %s with shape: %s' % (feed_name, kernel.shape))
        
    init = tf.constant_initializer(value=kernel, dtype=tf.float32)
    var = tf.get_variable(name="upscore_kernel", initializer=init, shape=kernel.shape)        
    return var

"""
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
"""


