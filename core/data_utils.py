"""Functions for loading and preprocessing dataset"""

from __future__ import print_function
import os
import gzip


def test():
    print('data_utils')

def load_vgg16_weight(path):

    """
        Dict keys:
        ['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3',        'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
    """
    # Initial network params
    # implement to load inital weight
    
    weight_dic = np.load(weight_dir, encoding='latin1').item()
    print(type(weight_dic))

    return data_dict
