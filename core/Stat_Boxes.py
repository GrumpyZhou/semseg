'''
This script computes the bounding boxes of each gt object, and return the statistics of their sizes.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import math
import sys
import os
import glob
from scipy import sparse
from scipy.misc import toimage
from scipy.misc import imsave
import matplotlib.pyplot as plt

#os.environ["CITYSCAPES_DATASET"] = "/Users/WY/Downloads/CityDatabase"
os.environ["CITYSCAPES_DATASET"] = "./data/CityDatabase"

MAX_instances = 30

def get_file_list(cityscapesPath):
    '''
    Give data path, find all .json files for gtFine
    '''

    searchFinetrain = os.path.join( cityscapesPath , "gtFine" , "train" , "*" , "*_gt*_mask.png")
    searchFineval = os.path.join( cityscapesPath , "gtFine" , "val" , "*" , "*_gt*_mask.png")

    filesFinetrain = glob.glob(searchFinetrain)
    filesFineval = glob.glob(searchFineval)
    filesFine = filesFinetrain + filesFineval
    filesFine.sort()

    if not filesFine:
        sys.exit('Did not find any files.')
    print('Got {} mask files. '.format(len(filesFine)))
    return filesFine

def open_gt_file(fname):
    #print('open file: ', fname)
    img = Image.open(fname)
    image = np.array(img, dtype=np.int8)
    (Height, Width, Channel) = np.shape(image)
    img_shape = (Height, Width)
    return image, img_shape

def cal_Object_Coord(img_mask, inst_num):
    '''
    Calculate an object's dimension.
    '''

    where = np.equal(img_mask, inst_num)
    indices = np.where(where)

    np_indices = np.array(indices)
    if not np_indices.size:
        return None, None

    max_x = indices[0].max()
    min_x = indices[0].min()
    max_y = indices[1].max()
    min_y = indices[1].min()
    len_x = max_x - min_x
    len_y = max_y - min_y

    return len_x, len_y

def get_Objects_of_Mask(fname):
    list_of_obj = []
    (image, img_shape) = open_gt_file(fname)

    for inst_num in range(MAX_instances):
        (len_x, len_y) = cal_Object_Coord(image[:,:,1], inst_num+1)
        if len_x is not None:
            list_of_obj.append([len_x, len_y])

    return list_of_obj

def get_statistics(files):
    x_stat = [0] * 108
    y_stat = [0] * 236
    for fname in files:
        list_of_objects = get_Objects_of_Mask(fname)
        for items in list_of_objects:
            # X length
            index_x = math.ceil(items[0] / 8.0 - 21.0)
            index_y = math.ceil(items[1] / 8.0 - 21.0)
            if index_x <= 0:
                x_stat[0] += 1
            else:
                x_stat[int(index_x)] += 1
            if index_y <= 0:
                y_stat[0] += 1
            else:
                y_stat[int(index_y)] += 1

    return x_stat, y_stat

def main():

    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']

    files = get_file_list(cityscapesPath)
    x_stat, y_stat = get_statistics(files)

    print('The x length stat:')
    print(x_stat)
    print('The y length stat:')
    print(y_stat)

if __name__ == "__main__":
    main()
