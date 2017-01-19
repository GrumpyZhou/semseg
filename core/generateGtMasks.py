'''
Give the following parameters:
cityscapesPath: default is './data/CityDatabase'
classnames: specify which class you want to segment with instance
            *IMPORTANT* if you change this, you have to modify label.py
                        and regenerate '*_gt*_instanceTrainIds.png' gt files.
MAX_instances: specify max number of instances of each class

Output: the corresponding ground truth masks for each '*_gt*_instanceTrainIds.png' gt file
e.g. input file:  aachen_000000_000019_gtFine_instanceTrainIds.png
     output file:  aachen_000000_000019_gtFine_mask.png

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
from scipy.misc import imsave

os.environ["CITYSCAPES_DATASET"] = "/Users/WY/Downloads/CityDatabase"

def get_file_list(cityscapesPath):
    '''
    Give data path, find all .json files for gtFine
    '''

    searchFinetrain = os.path.join( cityscapesPath , "gtFine" , "train" , "*" , "*_gt*_instanceTrainIds.png")
    searchFineval = os.path.join( cityscapesPath , "gtFine" , "val" , "*" , "*_gt*_instanceTrainIds.png")

    filesFinetrain = glob.glob(searchFinetrain)
    filesFineval = glob.glob(searchFineval)
    filesFine = filesFinetrain + filesFineval
    filesFine.sort()

    if not filesFine:
        sys.exit('Did not find any files.')
    print('Got {} instance files. '.format(len(filesFine)))
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
            label_id = int(label_id)
            # print('pixel {}, lable {}'.format(pixel, label_id))
            if pixel == 19: #Background
                # Ignore background
                continue
            elif label_id == classname[1]: #class_id
                inst_id = pixel % 1000
                if inst_id in instances[classname[0]]:
                    # remember the pixel's coordinates
                    x_coord = row
                    y_coord = col
                    instances[classname[0]][inst_id]['pixels'].append((x_coord, y_coord))
                else: # the first time seeing this instance id
                    instances[classname[0]][inst_id] = {}
                    # and remember its coordinates
                    x_coord = row
                    y_coord = col
                    instances[classname[0]][inst_id]['pixels'] = []
                    instances[classname[0]][inst_id]['pixels'].append((x_coord, y_coord))
                    instances[classname[0]][inst_id]['pixel_avg'] = [0.0, 0.0]
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
        for inst, values in instances[label].items():
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
        # print('sort_instance, class: {}'.format(label))
        class_avg_pixels[label] = []
        for inst, values in instances[label].items():
            class_avg_pixels[label].append((inst, values['pixel_avg'].tolist()))
    # sort the list
    for label in class_labels:
        class_avg_pixels[label] = sorted(class_avg_pixels[label], key=lambda tup: tup[1])

    return class_avg_pixels

def generate_sparse_mask(instances, class_avg_pixels, MAX_instances, img_shape):
    Gt_mask = {}
    Height = img_shape[0]
    Width = img_shape[1]
    class_labels = class_avg_pixels.keys()
    for label in class_labels:
        index = 0
        for index, item in enumerate(class_avg_pixels[label]):
            if index < MAX_instances:
                inst = item[0]
                pixel_array = np.array(instances[label][inst]['pixels'])
                row = pixel_array[:,0]
                col = pixel_array[:,1]
                fill_data = np.ones(len(row), dtype=np.int8) * (index + 1)
                data = sparse.coo_matrix((fill_data, (row, col)), shape=(Height, Width), dtype=np.int8).tocsc()
                if label in Gt_mask:
                    Gt_mask[label] += data
                else:
                    Gt_mask[label] = data
        # if lable is not in Gt_mask, generate 0 matrix
        # print('label is {}, after index is {}.'.format(label, index))
        if label not in Gt_mask:
            Gt_mask[label] = sparse.csc_matrix((Height, Width), dtype=np.int8)
    # Gt_mask_final
    # print('final assembly: ')
    for key in iter(Gt_mask):
        # Convert to full size matrix
        Gt_mask[key] = Gt_mask[key].toarray()
        # print('key: {}'.format(key))
    Gt_mask_final = np.dstack((Gt_mask['person'],Gt_mask['car']))
    # print('final mask shape: ', np.shape(Gt_mask_final))

    return Gt_mask_final
def generate_masks(instances, class_avg_pixels, MAX_instances, img_shape):
    '''
    Generate masks for the Ground truth;
    MAX_instances: specify the max number of instances for each class
    Return: stacked masks, car comes first, then person for now
    '''
    Gt_mask = {}
    Height = img_shape[0]
    Width = img_shape[1]
    class_labels = class_avg_pixels.keys()
    print('in generate, the class_labels are: {}'.format(class_labels))
    for label in class_labels:
        index = 0
        for index, item in enumerate(class_avg_pixels[label]):
            if index < MAX_instances:
                inst = item[0]
                pixel_array = np.array(instances[label][inst]['pixels'])
                row = pixel_array[:,0]
                col = pixel_array[:,1]
                fill_data = np.ones(len(row), dtype=np.int8)
                # mask = sparse.coo_matrix((fill_data, (row, col)), shape=(Height, Width), dtype=np.int8).toarray()
                mask = sparse.coo_matrix((fill_data, (row, col)), shape=(Height, Width), dtype=np.int8)
                if label in Gt_mask:
                    # Gt_mask[label] = np.dstack((Gt_mask[label], mask))
                    Gt_mask[label].append(mask)
                    # print('shape of gt_label {} is {}'.format(label, np.shape(Gt_mask[label])))
                else:
                    # Gt_mask[label] = mask
                    Gt_mask[label] = []
                    Gt_mask[label].append(mask)
        # fill the remaining masks with zeros
        if index < MAX_instances - 1:
            if index == 0:
                remaining = MAX_instances
            else:
                remaining = MAX_instances - index - 1
            mask = np.zeros((Height, Width), dtype=np.int8)
            for i in range(remaining):
                # need to check if there exists such an instance of this class
                if label in Gt_mask:
                    # Gt_mask[label] = np.dstack((Gt_mask[label], mask))
                    Gt_mask[label].append(mask)
                    # print('shape of gt_label {} is {}'.format(label, np.shape(Gt_mask[label])))
                else:
                    # Gt_mask[label] = mask
                    Gt_mask[label] = []
                    Gt_mask[label].append(mask)

    # The final masks of ground truth
    # print('final assembly: ')
    # for key in iter(Gt_mask):
    #     print('key: {}'.format(key))
    # Gt_mask_final = np.dstack((Gt_mask['car'],Gt_mask['person']))
    Gt_mask_final = Gt_mask['car'] + Gt_mask['person']

    return Gt_mask_final

def main():

    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']

    instances = {}
    classnames = [('car', 13), ('person', 11)]
    MAX_instances = 30
    files = get_file_list(cityscapesPath)
    # files = ['/Users/WY/Desktop/instance-data/aachen_000004_000019_gtFine_instanceTrainIds.png']

    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ))

    for fname in files:
        # image is np.array, dtype=np.int16, has a shape of img_shape
        (image, img_shape) = open_gt_file(fname)
        # print('open file {}, shape {}'.format(fname, img_shape))
        for classname in classnames:
            instances[classname[0]] = {}
            create_instance_data(instances, classname, image, img_shape)
        cal_pixel_avg(instances)
        class_avg_pixels = sort_instances(instances)
        # print('in main, class labels are {}'.format(class_avg_pixels.keys()))
        # Gt_mask = generate_masks(instances, class_avg_pixels, MAX_instances, img_shape)
        Gt_mask = generate_sparse_mask(instances, class_avg_pixels, MAX_instances, img_shape)
        # fname = fname.replace('png', 'npy')
        fname = fname.replace('instanceTrainIds', 'mask')
        # fname = fname.replace('png', 'pickle')
        # np.save(fname, Gt_mask)
        height= np.shape(Gt_mask)[0]
        width = np.shape(Gt_mask)[1]
        stacked = np.zeros((height, width), dtype=np.int8)
        save_mask = np.dstack((Gt_mask, stacked))
        # print('shape of saved is {}'.format(np.shape(save_mask)))
        print('Save gt mask to {}.'.format(fname))
        # imsave(fname, save_mask)
        toimage(save_mask, high=29, low=0, cmin=0, cmax=29).save(fname)
        # cPickle.dump(Gt_mask, open(fname, "w"))

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
