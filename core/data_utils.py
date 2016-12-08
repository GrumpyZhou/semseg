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

        self.indices = self.load_indices(params.get('dataset', 'train'), 
                                         params.get('classes', None),
                                         params.get('filter_no_label',False))
        self.idx = 0
        # make eval deterministic
        if 'train' not in params['dataset']:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def next_batch(self, predef_inx=None):
        """
        - Reshape image and label, extend 1st axis for batch dimension
        - If 'predef_inx' is given, load sepecific image, 
          Otherwise load randomly selected(if self.random is set), or incrementally
        - Return: (image, label)
        """
        if predef_inx is None:
            # pick next input
            if self.random:
                self.idx = random.randint(0, len(self.indices)-1)
            else:
                self.idx += 1
                if self.idx == len(self.indices):
                    self.idx = 0
            idx_str = self.indices[self.idx]
        else:
            idx_str = predef_inx
            
        print('Batch index string: %s'% idx_str)
        image = self.load_image(idx_str)
        image = image.reshape(1, *image.shape) 
        label = self.load_label(idx_str)
        if label is not None:
            label = label.reshape(1, *label.shape)
            
        return (image,label)
   
    def load_indices(self, fold_type='train', classes_dict=None, filter_no_label=False):
        """ 
        Load indices of images and labels as list
        - fold_type: train, val, trainval
        - class_name: predefined classes of the dataset
        - filter_no_label: filter all indices that have no ground truth
        """
        if filter_no_label or classes_dict is None:
            idx_dir = os.path.join(self.voc_dir,'ImageSets/Segmentation/trainval.txt')
            with open(idx_dir, 'rb') as f: 
                default_indices = f.read().splitlines()
        
        if classes_dict is None: 
            # Load from default segmentation dataset
            indices = default_indices
            print('Load indices from %s : %d' %(idx_dir,len(indices)))
            
        else:
            indices = []
            for class_name in classes_dict:
                if class_name not in self.classes:
                    print('Invalid class name %s!'% class_name)
                    sys.exit()
                else:
                    # Load specified class indices
                    idx_dir = os.path.join(self.voc_dir,'ImageSets/Main','%s_%s.txt'%(class_name,fold_type))
                    with open(idx_dir, 'rb') as f: 
                        indices_ = f.read().splitlines()
                        for i in range(len(indices_)):
                            idx= indices_[i]
                            indices_[i] = indices_[i].split(' ')[0]
                
                    if filter_no_label:
                        #list(set(default_indices).intersection(indices)) work as well
                        indices_ = filter(lambda x:x in default_indices, indices_)
                    print('Load indices from %s : %d' %(idx_dir,len(indices_)))
                    indices += indices_
                    
            print('total indices:%d'% len(indices))

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
        
        try:
            img = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        except IOError as e:
            print('Warning: no label with index : %s!!'%idx)
            label = None
            return label

        label = np.array(img, dtype=np.uint8)
        label = label[np.newaxis, ...]

        return label
'''
#Testing example
params = {'voc_dir':"data/VOC2012",
          'dataset':'train',
          'classes':['person','cat','dog'], # All classes are loaded if class is None
          'filter_no_label':True,         # Filter all indices with no ground truth, default is False
          'randomize': True,
          'seed': None}

dt = VOCDataSet(params)
data = dt.next_batch()
'''

def load_vgg16_weight(path):

    # Initial network params
    fpath = os.path.abspath(os.path.join(path, os.curdir))
    data_dict = np.load(fpath, encoding='latin1').item()
    print("Successfullt load vgg16 weight from %s."%fpath)
    return data_dict

def color_image(image, num_classes=22):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))



