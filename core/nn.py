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

def conv_layer(x, feed_dict, name, stride=1, var_dict=None):

    with tf.variable_scope(name) as scope:

        kernel = get_conv_kernel(feed_dict, name)
        conv = tf.nn.conv2d(x, kernel,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        bias = get_bias(feed_dict, name)
        conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)

    if var_dict is not None:
        var_dict[name] = (kernel, bias)

    return conv_out

def fully_conv_layer(x, feed_dict, name, shape, relu=True, dropout=False, keep_prob=0.5, random=False, var_dict=None):
    with tf.variable_scope(name) as scope:
        
        kernel = get_fconv_weight(feed_dict, name, shape, random=random)
        conv = tf.nn.conv2d(x, kernel,
                            strides = [1, 1, 1, 1],
                            padding = 'SAME')
        bias = get_bias(feed_dict, name, shape=[shape[3]], random=random)
        print("bias shape, fully conv %s: %s" % (name, bias.get_shape()))
        print("kernel shape, fully conv %s: %s" % (name, kernel.get_shape()))
        conv_out = tf.nn.bias_add(conv, bias)

        if relu:
            conv_out =  tf.nn.relu(conv_out)
        if dropout:
            conv_out = tf.nn.dropout(conv_out, keep_prob)

    if var_dict is not None:
        var_dict[name] = (kernel, bias)

    return conv_out


def score_layer(x, name, num_classes, random=True, stddev=0.001, feed_dict=None, var_dict=None):
    '''
    Note: use random=True only when training!
    if random=False, load trained weights on this layer
    '''
    # Use random kernel for convolution to calculate the score
    with tf.variable_scope(name) as scope:
        if random:  # if use random kernel to calculate score
            in_features = x.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            with tf.variable_scope(name) as scope:
                init_w = tf.truncated_normal_initializer(stddev=stddev)
                weight = tf.get_variable(name='weight', shape=shape, initializer=init_w)
                conv = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')

                init_b = tf.constant_initializer(0.0)
                bias = tf.get_variable(name="bias", initializer=init_b, shape=[num_classes])
                # save kernel var
                score = tf.nn.bias_add(conv, bias)

                print("score layer, weights: %s" % weight.get_shape())
                print("score layer, bias: %s" % bias.get_shape())

                if var_dict is not None: 
                    var_dict[name] = (weight, bias)

        else:   # Don't use random kernel, use trained weights
            # name = 'fc8'  # the name used in VGG16-net
            # we use the name = 'score_fr', trained by our network
            if not feed_dict.has_key(name):
                print("Weight dataset has no name: ", name)
                print("Failed loading weights from score(fc8) layer!")

                # TODO: load weights from VGG16-net
                name = 'fc8'
                shape = [1,1,4096, 1000]
                score = fully_conv_layer(x, feed_dict, name, shape, relu=False, random=random, var_dict=var_dict)

            else:
                shape = [1,1,4096, 22]
                score = fully_conv_layer(x, feed_dict, name, shape, relu=False, random=random, var_dict=var_dict)


    return score

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
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
                                        strides=strides, padding='SAME')

    return deconv

def get_conv_kernel(feed_dict, name):
    if not feed_dict.has_key(name):
        print("Weights databast has no name: ", name)
    kernel = feed_dict[name][0]
    shape = kernel.shape
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % str(shape))

    init = tf.constant_initializer(value=kernel,dtype=tf.float32)
    var = tf.get_variable(name="kernel", initializer=init, shape=shape)
    return var

def get_bias(feed_dict, name, shape=None, random=False):
    if not random:
        if not feed_dict.has_key(name):
            print("Feed_dict doesn't contain key:%s, initialize a random bias", name)
            init = tf.constant_initializer(0.1, dtype=tf.float32)
        else:
            bias = feed_dict[name][1]
            init = tf.constant_initializer(value=bias, dtype=tf.float32)        
            shape = bias.shape
            #print('Layer name: %s' % name)
            #print('Layer bias shape: %s' % str(shape))
    else:
        init = tf.constant_initializer(0.1, dtype=tf.float32)
        
    var = tf.get_variable(name="bias", initializer=init, shape=shape)
    return var

def get_fconv_weight(feed_dict, name, shape, num_class=None, random=False):
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % shape)
    if not random:
        if not feed_dict.has_key(name):
            print("Feed_dict doesn't contain key:%s, initialize a random weight", name)
            init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
        weights = feed_dict[name][0]
        weights = weights.reshape(shape)
        init = tf.constant_initializer(value=weights,
                                    dtype=tf.float32)
    else:
        init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
    var = tf.get_variable(name="weight", initializer=init, shape=shape)
    return var

def get_deconv_filter(f_shape):
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
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)

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


