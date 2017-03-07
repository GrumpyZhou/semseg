from __future__ import print_function

from PIL import Image
import os
import sys
import random
import numpy as np

from dataset.VOCDataSet import VOCDataSet
from dataset.CityDataSet import CityDataSet

def load_weight(path):

    # Initial network params
    fpath = os.path.abspath(os.path.join(path, os.curdir))
    data_dict = np.load(fpath, encoding='latin1').item()
    print("Successfully load weight file from %s."%fpath)
    return data_dict

def vgg16_weight_transform(vgg16_path, vgg16_new_path):
    '''
    This function is used to transform the format for original vgg16.npy 
    to fit into our network structure.
    Given the path of vgg16.npy file, the transformed weight file is saved to 
    specified path.
    '''
    # Load old weight dict
    data_dict = load_weight(vgg16_path)
    # Remove weight of fc layers
    data_dict.pop('fc6', None)
    data_dict.pop('fc8', None)
    # Expand dimension of fc7 filter
    data_dict['fc7'][0] = np.reshape(data_dict['fc7'][0],(1,1,4096,4096))

    # Add conv6 layers using weight of conv5 layers and conv7
    dict_to_add = {'conv6_1': data_dict['conv5_1'],
                   'conv6_2': data_dict['conv5_2'],
		   'conv7': data_dict.pop('fc7', None)}
    data_dict.update(dict_to_add)
    print('New keys:%s'%str(data_dict.keys()))
    # Save result
    np.save(vgg16_new_path, data_dict)
    print("Successfully save weight file to %s."%vgg16_new_path)

def temp_weight_transform(path, new_path):
    '''To adapt already generated .npy e.g city_fcn32.npy city_fcn16_skip.npy
       will be deleted later'''
    data_dict = load_weight(path)
    data_dict['conv6_1'] = data_dict.pop('fc6_1', None)
    data_dict['conv6_2'] = data_dict.pop('fc6_2', None)
    data_dict['conv6_3'] = data_dict.pop('fc6_3', None)
    data_dict['conv7'] = data_dict.pop('fc7', None)
    print('New keys:%s'%str(data_dict.keys()))

    np.save(new_path, data_dict)
    print("Successfully save weight file to %s."%new_path)

 

# Deprecated because produced color truth image does not match the original
def color_image(image, num_classes=22):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

