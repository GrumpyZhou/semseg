"""Functions for loading and preprocessing dataset"""

from __future__ import print_function
import os
import sys
import gzip
import numpy as np


def test():
    print('data_utils')


def load_vgg16_weight(path):

    # Initial network params
    path = os.path.abspath(os.path.join(path, os.curdir))
    fpath = os.path.join(path, "vgg16.npy")
    
    data_dict = np.load(fpath, encoding='latin1').item()
    print("Successfullt load vgg16 weight from %s."%fpath)
    print("Data dict keys:\n",data_dict.keys())
    return data_dict

