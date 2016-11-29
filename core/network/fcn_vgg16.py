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
import data_utils

data_utils.test()
nn.test()
class FCN16VGG:

    def __init__(self, data_dir=None):
        """Dict keys:['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3','conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']"""
        # Load VGG16 pretrained weight
        data_dict = dt.load_vgg16_weight(data_dir)
        self.data_dict = data_dict

        # Init other necessary parameters

    def inference(self, image ):
        # Image preprocess
        # Network structure -- VGG16
        # Upsampling
        # Return predict
        pass

    def train(self, total_loss, learning_rate ):
        # To be implemented Later
        # Mini-batch
        # Minimize loss
        # Add necessary params to summary
        # Return train_op
        pass
