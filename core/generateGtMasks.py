'''
Give the following parameters:
cityscapesPath: default is './data/CityDatabase'
classnames: specify which class you want to segment with instance
            *IMPORTANT* if you change this, you have to modify label.py
                        and regenerate '*_gt*_instanceTrainIds.png' gt files.
MAX_instances: specify max number of instances of each class

Output: the corresponding ground truth masks for each '*_gt*_instanceTrainIds.png' gt file
e.g. input file:  aachen_000000_000019_gtFine_instanceTrainIds.png
     output file:  aachen_000000_000019_gtFine_instanceTrainIds.npy

*NOTE* The output file is a full size matrix, not sparse!
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

def get_file_list(cityscapesPath):
    '''
    Give data path, find all *_gt*_instanceTrainIds.png files for gtFine
    '''
    searchFine = os.path.join( cityscapesPath , "gtFine" , "*" , "*" , "*_gt*_instanceTrainIds.png" )
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    if not files:
        sys.exit('Did not find any files.')
    return filesFine

def open_gt_file(fname):
    img = Image.open(fname)
    image = np.array(img, dtype=np.int16)
    (Height, Width) = np.shape(image)
    img_shape = (Height, Width)

    return image, img_shape

def create_instance_data(instances, classname, image, img_shape):
    '''
    For given image and the classname it contains,
    create corresponding instance data.
    instances: Dict
    classname: (class_label, class_id)
    image: 1024*2048 np.array
    img_shape: (Height, Width)
    '''
    Height = img_shape[0]
    Width = img_shape[1]
    for row in range(Height):
        for col in range(Width):
            pixel = image[row][col]
            label_id = pixel / 1000
            if pixel == 19: #Background
                # Ignore background
                continue
            elif label_id == classname[1]: #class_id
                inst_id = pixel % 1000
                if inst_id in instances[classname]:
                    # remember the pixel's coordinates
                    x_coord = row
                    y_coord = col
                    instances[classname][inst_id]['pixels'].append((x_coord, y_coord))
                else: # the first time seeing this instance id
                    instances[classname][inst_id] = {}
                    # and remember its coordinates
                    x_coord = row
                    y_coord = col
                    instances[classname][inst_id]['pixels'].append((x_coord, y_coord))
                    instances[classname][inst_id]['pixel_avg'] = [0.0, 0.0]
            else:
                # Ignore othere values
                continue

def cal_pixel_avg(instances):
    '''
    Compute pixel average coordinates (x_avg, y_avg)
    for every instance in an image
    '''
    class_labels = instances.keys()
    for label in class_labels:
        for inst, values in instances[label]:
        coord_avg = np.mean(values['pixels'], axis=0)
        instances[label][inst]['pixel_avg'] = coord_avg

def sort_instances(instances):
    '''
    Build a list for each class where each item is: (inst, [x_avg, y_avg])
    Return: ordered instances for each class
    '''
    class_avg_pixels = {}
    class_labels = instances.keys()
    # extract pixel_avg to a list
    for label in class_labels:
        class_avg_pixels[label] = []
            for inst, values in instances[label].items():
                class_avg_pixels[label].append((inst, values['pixel_avg']))
    # sort the list
    for label in class_labels:
        class_avg_pixels[label] = sorted(class_avg_pixels[label], key=lambda tup: tup[1])

    return class_avg_pixels

def generate_masks(class_avg_pixels, MAX_instances, img_shape):
    '''
    Generate masks for the Ground truth;
    MAX_instances: specify the max number of instances for each class
    Return: stacked masks, car comes first, then person for now
    '''
    Gt_mask = {}
    Height = img_shape[0]
    Width = img_shape[1]
    class_labels = class_avg_pixels.keys()
    for label in class_labels:
        for index, item in enumerate(class_avg_pixels[label]):
            if index < MAX_instances:
                inst = item[0]
                pixel_array = np.array(instances[label][inst]['pixels'])
                row = pixel_array[:,0]
                col = pixel_array[:,1]
                fill_data = np.ones(len(row), dtype=np.int8)
                mask = sparse.coo_matrix((fill_data, (row, col)), shape=(Height, Width)).toarray()
                if label in Gt_mask:
                    Gt_mask[label] = np.stack((Gt_mask[label], mask))
                else:
                    Gt_mask[label] = mask
        # fill the remaining masks with zeros
        if index < MAX_instances - 1:
            remaining = MAX_instances - index - 1
            mask = np.zeros((Height, Width), dtype=np.int8)
            for i in range(remaining):
                # need to check if there exists such an instance of this class
                if label in Gt_mask:
                    Gt_mask[label] = np.stack((Gt_mask[label], mask))
                else:
                    Gt_mask[label] = mask

    # The final masks of ground truth
    Gt_mask = np.stack((Gt_mask['car'],Gt_mask['person']))

    return Gt_mask

def main(argv):
    cityscapesPath = './data/CityDatabase'
    instances = {}
    classnames = [('car', 13), ('person', 11)]
    MAX_instances = 100
    files = get_file_list()

    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')

    for fname in files:
        # image is np.array, dtype=np.int16, has a shape of img_shape
        (image, img_shape) = open_gt_file(fname)
        for classname in classnames:
            create_instance_data(instances, classname, image, img_shape)
        cal_pixel_avg(instances)
        class_avg_pixels = sort_instances(instances)
        Gt_mask = generate_masks(class_avg_pixels, MAX_instances, img_shape)
        fname = fname.replace('png', 'npy')
        np.save(fname, Gt_mask)

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()

if __name__ == "__main__":
    main()
