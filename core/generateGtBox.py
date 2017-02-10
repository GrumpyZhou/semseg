'''
This script computes the bounding boxes of each gt object,
and return the upper-left corner of the box.
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

import Stat_Boxes

MAX_instances = 30
Box_Width = 392
Box_Height = 448

def cal_precise_positive(img_mask, inst_num):
    '''
    Give precise coordinate of positive box.
    '''

    where = np.equal(img_mask, inst_num)
    indices = np.where(where)

    np_indices = np.array(indices)
    if not np_indices.size:
        return None, None

    min_x = indices[0].min()
    min_y = indices[1].min()

    return min_x, min_y

def get_precise_posiboxes_of_Mask(fname):
    list_of_preposiboxes = []
    img, img_shape = Stat_Boxes.open_gt_file(fname)

    for inst_num in range(MAX_instances):
    	(x_coord, y_coord) = cal_precise_positive(img[:,:,1], inst_num+1)
        if x_coord is not None:
            list_of_preposiboxes.append([x_coord, y_coord])

    return list_of_preposiboxes


def is_positive(precise_pos_list, test_coord):
    '''
    for the test_coord, test if it is a positive box
    '''
    for precise_coord in precise_pos_list:
        if abs(precise_coord[0] - test_coord[0]) <= 16 and abs(precise_coord[1] - test_coord[1]) <= 16:
            # The test_coord is a positive box
            return True

    return False

def random_pair(rand_type):
    '''
    rand_type: positive, negative
    '''
    if rand_type == 'positive':
        rand_num = np.random.randint(-16,16)
    elif rand_type == 'negative':
        rand_num = np.random.randint(-448,448)
        if rand_num >= 0 and rand_num <= 16:
            rand_num += 16
        elif rand_num <= 0 and rand_num >= -16:
            rand_num -= 16
    else:
        sys.exit('Invalid rand_type. Must be positive or negative')

    return rand_num


def generate_box(fname):
	box_data = {}
	box_data['positive'] = []
	box_data['negative'] = []

	list_of_preposiboxes = get_precise_posiboxes_of_Mask(fname)
	number_of_preposiboxes = len(list_of_preposiboxes)
	loop_num = int(128 / number_of_preposiboxes)
	loop_remainder = int(128 % number_of_preposiboxes)
	# Generating 128 positive and negative boxes, respectively.
	for index in range(loop_num):
		for pre_coord in list_of_preposiboxes:
			# Generate positive box
			posi_box = generate_positive_box(pre_coord)
			box_data['positive'].append(posi_box)

			# Generate negative box
			nega_box = generate_negative_box(pre_coord)
			# If the negative box is not overlap with another positive box
			while is_positive(list_of_preposiboxes, nega_box):
				nega_box = generate_negative_box(pre_coord)
			box_data['negative'].append(nega_box)

	# Generate for the remainder boxes.
	for pre_coord in list_of_preposiboxes:
		if loop_remainder == 0:
			break
		# Generate positive box
		posi_box = generate_positive_box(pre_coord)
		box_data['positive'].append(posi_box)

		# Generate negative box
		nega_box = generate_negative_box(pre_coord)
		# If the negative box is not overlap with another positive box
		while is_positive(list_of_preposiboxes, nega_box):
			nega_box = generate_negative_box(pre_coord)
		box_data['negative'].append(nega_box)
		loop_remainder -= 1

		return box_data

def generate_positive_box(precise_posi_coord):
	'''
	Compute the randomized upper left corner of positive box.
	'''
	rand_shift = random_pair('positive')
	temp_x = precise_posi_coord[0] + rand_shift
	temp_y = precise_posi_coord[1] + rand_shift

	# Boundary of y: 1024-448-1
	if temp_y > 575 or temp_y < 0:
		# shift to the other direction
		temp_y = precise_posi_coord[1] - rand_shift
	# Boundary of x: 2048-392-1
	if temp_x > 1655 or temp_x < 0:
		temp_x = precise_posi_coord[0] - rand_shift

	return (temp_x, temp_y)


def generate_negative_box(precise_posi_coord):
	'''
	Compute the randomized upper left corner of negative box.
	'''
	rand_shift = random_pair('negative')
	temp_x = precise_posi_coord[0] + rand_shift
	temp_y = precise_posi_coord[1] + rand_shift

	# Boundary of y: 1024-448-1
	if temp_y > 575 or temp_y < 0:
		# shift to the other direction
		temp_y = precise_posi_coord[1] - rand_shift
	# Boundary of x: 2048-392-1
	if temp_x > 1655 or temp_x < 0:
		temp_x = precise_posi_coord[0] - rand_shift

	return (temp_x, temp_y)

def main():

    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
    	sys.exit('No data path found')

    files = Stat_Boxes.get_file_list(cityscapesPath)
    for fname in files:
    	box_data = generate_box(fname)
    	fname = fname.replace('mask.png', 'box.npy')
    	np.save(fname, box_data)
    	print('write box file to %s'%fname)

if __name__ == "__main__":
    main()
