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

# for testing
import data_utils as dt

def test():
    data_dict = dt.load_vgg16_weight('data')
    name = 'fc6'
    print(name,data_dict[name][0].shape)
    name = 'fc7'
    print(name,data_dict[name][0].shape)
    name = 'fc8'
    print(name,data_dict[name][0].shape)

test()

"""
 Dict keys:
['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
"""

def max_pool_layer(x, name, stride=2):
    pool = tf.nn.max_pool(x, ksize=[1, stride, stride, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME', name=name)
    return pool

def conv_layer(x, feed_dict, name, stride=1):

    with tf.variable_scope(name) as scope:

        kernel = get_conv_kernel(feed_dict, name)
        conv = tf.nn.conv2d(x, kernel,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        bias = get_bias(feed_dict, name)
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
        return conv_out


def fully_conv_layer(x, feed_dict, name, shape, relu=True, dropout=False, keep_prob=0.5):
    #print('!!!!!', name)
    with tf.variable_scope(name) as scope:
        kernel = get_fconv_weight(feed_dict, name, shape)
        conv = tf.nn.conv2d(x, kernel,
                            strides = [1, 1, 1, 1],
                            padding = 'SAME')
        bias = get_bias(feed_dict, name)
        print("bias shape, fully conv %s: %s" % (name, bias.get_shape()))
        print("kernel shape, fully conv %s: %s" % (name, kernel.get_shape()))
        conv_out = tf.nn.bias_add(conv, bias)

        if relu:
            conv_out =  tf.nn.relu(conv_out)
        if dropout:
            conv_out = tf.nn.drop(conv_out, keep_prob)
        return conv_out


def score_layer(x, name, num_classes, random=True, stddev=0.001, feed_dict=None):
    # Use random kernel for convolution to calculate the score
    #num_class = shape[3]
    with tf.variable_scope(name) as scope:
        if random:  # if use random kernel to calculate score
            in_features = x.get_shape()[3].value
            print("in_feature, %d" % in_features)
            shape = [1, 1, in_features, num_classes]
            print("num_classes, %d" % num_classes)
            with tf.variable_scope(name) as scope:
                init_w = tf.truncated_normal_initializer(stddev=stddev)
                #print()
                weight = tf.get_variable(name='weight', shape=shape, initializer=init_w)
                conv = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')

                init_b = tf.constant_initializer(0.0)
                bias = tf.get_variable(name="bias", initializer=init_b, shape=[num_classes])
                score = tf.nn.bias_add(conv, bias)

                print("score layer, weights: %s" % weight.get_shape())
                print("score layer, bias: %s" % bias.get_shape())
        else:   # Don't use random kernel, use VGG16 fc_weights
            name = 'fc8'
            shape = [1,1,4096,1000]     # Assume 1000 way (classes)
            score = fully_conv_layer(x, feed_dict, name, shape, relu=False)

    return score

# def upscore_layer(x, feed_dict, name, ksize=4, stride=2):
# Redefine, previous definition was inconsistent.
'''
def upscore_layer(x, name, shape, ksize=4, stride=2):
    # WY
    # feed_dict here is unnecessary, since upsampling params are fixed to biliner interpolation.
    # Get original input image size
    new_height = shape[1]
    new_width = shape[2]
    size = [new_height, new_width]

    # Create upsampled prediction
    upscore = tf.image.resize_bilinear(x, size)
    return upscore
'''

# Use existing code, still don't understand. Prefer to use upscore_layer() first.
def upscore_layer(x, name, shape, num_class, ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
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

        weights = get_deconv_filter(f_shape)
        #print("deconv result weights: %s" % weights.get_shape())
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
                                        strides=strides, padding='SAME')

    return deconv

def get_conv_kernel(feed_dict, name):
    print(name)
    kernel = feed_dict[name][0]
    shape = kernel.shape
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))

    init = tf.constant_initializer(value=kernel,dtype=tf.float32)
    var = tf.get_variable(name="kernel", initializer=init, shape=shape)
    return var

def get_bias(feed_dict, name):
    bias = feed_dict[name][1]
    shape = bias.shape
    print('Layer name: %s' % name)
    print('Layer bias shape: %s' % str(shape))

    init = tf.constant_initializer(value=bias, dtype=tf.float32)
    var = tf.get_variable(name="bias", initializer=init, shape=shape)
    return var

def get_fconv_weight(feed_dict, name, shape, num_class=None):
    # size = shape[0] * shape[1] * shape[2]
    # weight = feed_dict[name][0]
    # if size == tf.shape(weight)[0]:
    #     weight = weight.reshape(shape)
    #     shape = weight.shape
    #     init = tf.constant_initializer(value=weight, dtype=tf.float32)
    #     print('Layer name: %s' % name)
    #     print('Layer shape: %s' % str(shape))
    # else:
    #     print('Layer %s shape not matching, initial a new one.' % name)
    #     init = tf.truncated_normal_initializer(stddev=0.1)

    # var = tf.get_variable(name="weight", initializer=init, shape=shape)
    print('Layer name: %s' % name)
    print('Layer shape: %s' % shape)
    weights = feed_dict[name][0]
    weights = weights.reshape(shape)
    init = tf.constant_initializer(value=weights,
                                    dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    return var

def get_deconv_filter(f_shape):
    # WY
    # Bilinear interpolation
    '''
    To compute deconv filter weights.
    Since weights are fixed to biliner interpolation,
    use tf.image.resize_bilinear() directly in upscore_layer()

    This function is only used for _upscore_layer(),
    if upscore_layer() is used, then this function will not be invoked
    '''
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)
    #return None

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


