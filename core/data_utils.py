"""Functions for loading and preprocessing dataset"""

from __future__ import print_function

from PIL import Image
import os
import sys
import random
import numpy as np


class VOCDataSet():

    def __init__(self, params):
        # Root directory of VOC dataset
        self.voc_dir = params['voc_dir']
        self.mean = np.array((104.007, 116.669, 122.679), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        # Predefined classes
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.indices = self.load_indices(params['dataset'])
        self.idx = 0
        # make eval deterministic
        if 'train' not in params['dataset']:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def next_batch(self):
        """
        Processing data:
        - Random index selection(if set)
        - Reshape image and label, extend 1st axis for batch dimension
        - Return: (image, label)
        """

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        print('Batch index: %d'% self.idx)

        image = self.load_image(self.indices[self.idx])
        label = self.load_label(self.indices[self.idx])

        image = image.reshape(1, *image.shape)
        label = label.reshape(1, *label.shape)

        return (image,label)


    def load_indices(self, fname):
        """ Load indices of images and labels as list """
        idx_dir = os.path.join(self.voc_dir,'ImageSets/Segmentation','%s.txt'%fname)
        with open(idx_dir, 'rb') as f:
            indices = f.read().splitlines()
        print('Indices loaded: %d' %len(indices))
        return indices

    def load_image(self, idx):
        """
        Load input image and preprocess for using pretrained weight from Caffee:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        img = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        image = np.array(img, dtype=np.float32)
        image = image[:,:,::-1]     # RGB -> BGR
        image -= self.mean
        #image = image.transpose((2,0,1))
        return image

    def load_label(self,idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        img = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        label = np.array(img, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

#Testing example
# params = {'voc_dir':"data/VOCdevkit/VOC2012",
#           'dataset':'val',
#           'randomize': True,
#           'seed': None}

# dt = VOCDataSet(params)
# data = dt.next_batch()
# print(data[0].shape, data[1].shape)




def load_vgg16_weight(path):

    # Initial network params
    path = os.path.abspath(os.path.join(path, os.curdir))
    fpath = os.path.join(path, "vgg16.npy")

    data_dict = np.load(fpath, encoding='latin1').item()
    print("Successfullt load vgg16 weight from %s."%fpath)
    return data_dict

def color_image(image, num_classes=20):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))



