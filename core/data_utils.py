"""Functions for loading and preprocessing dataset"""

from __future__ import print_function

from PIL import Image
import os
import sys
import random
import numpy as np
import glob
from collections import namedtuple
from scipy.misc import imsave

# define a data structure
Label_City = namedtuple( 'Label' , ['name', 'trainId', 'color',] )

class CityDataSet():

    def __init__(self, params):
        self.city_dir = params['city_dir']
        self.cities_train = ['aachen','bochum','bremen','cologne','darmstadt',
                            'dusseldorf','erfurt','hamburg','hanover','jena',
                            'krefeld','monchengladbach','strasbourg','stuttgart',
                            'tubingen','ulm','weimar','zurich']
        self.cities_val = ['frankfurt','lindau','munster']
        (self.img_indices, self.lbl_indices) = self.load_indicies()

        # Create mapping of (lable_name, id, color)
        self.labels = [
            Label_City(  'road'          ,   0, (128, 64,128) ),
            Label_City(  'sidewalk'      ,   1, (244, 35,232) ),
            Label_City(  'building'      ,   2, ( 70, 70, 70) ),
            Label_City(  'wall'          ,   3, (102,102,156) ),
            Label_City(  'fence'         ,   4, (190,153,153) ),
            Label_City(  'pole'          ,   5, (153,153,153) ),
            Label_City(  'traffic light' ,   6, (250,170, 30) ),
            Label_City(  'traffic sign'  ,   7, (220,220,  0) ),
            Label_City(  'vegetation'    ,   8, (107,142, 35) ),
            Label_City(  'terrain'       ,   9, (152,251,152) ),
            Label_City(  'sky'           ,  10, ( 70,130,180) ),
            Label_City(  'person'        ,  11, (220, 20, 60) ),
            Label_City(  'rider'         ,  12, (255,  0,  0) ),
            Label_City(  'car'           ,  13, (  0,  0,142) ),
            Label_City(  'truck'         ,  14, (  0,  0, 70) ),
            Label_City(  'bus'           ,  15, (  0, 60,100) ),
            Label_City(  'train'         ,  16, (  0, 80,100) ),
            Label_City(  'motorcycle'    ,  17, (  0,  0,230) ),
            Label_City(  'bicycle'       ,  18, (119, 11, 32) ),
            Label_City(  'void'          ,  19, (  0,  0,  0) )
        ]
        self.trainId2Color = [label.color for label in self.labels]

        # Random params
        self.idx = 0
        self.random = params.get('randomize',True)
        self.seed = params.get('seed',None)

        # Randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.img_indices)-1) # random init

    def load_indicies(self):
        datasets = ['train','val']
        files_img = []
        files_lbl = []
        for ds in datasets:
            # Load training images
            search_img = os.path.join(self.city_dir,
                                      'leftImg8bit_trainvaltest/leftImg8bit',
                                      ds,'*','*_leftImg8bit.png')
            files_img += glob.glob(search_img)
            files_img.sort()

            # Load groudtruth images
            search_lbl = os.path.join(self.city_dir,
                                      'gtFine',
                                      ds,'*','*_gtFine_labelTrainIds.png')
            files_lbl += glob.glob(search_lbl)
            files_lbl.sort()

        print('Training images:%d Ground Truth images:%d',len(files_img), len(files_lbl))
        return (files_img, files_lbl)

    def next_batch(self):
        """
        - Reshape image and label, extend 1st axis for batch dimension
        - If 'predef_inx' is given, load sepecific image,
          Otherwise load randomly selected(if self.random is set), or incrementally
        - Return: (image, label)
        """
        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.img_indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.img_indices):
                self.idx = 0
        img_fname = self.img_indices[self.idx]
        lbl_fname = self.lbl_indices[self.idx]

        print('Batch index: %d '%self.idx)
        image = self.load_image(img_fname)
        image = image.reshape(1, *image.shape)
        label = self.load_label(lbl_fname)
        if label is not None:
            label = label.reshape(1, *label.shape)

        return (image,label)

    def load_image(self, fname):
        """
        Load input image and preprocess for using pretrained weight from Caffee:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print('Loading img:%s'%fname)
        try:
            img = Image.open(fname)
        except IOError as e:
            print('Warning: no image with name %s!!'%fname)

        image = np.array(img, dtype=np.float32)
        image = image[:,:,::-1]     # RGB -> BGR
        #image -= self.mean
        #image = image.transpose((2,0,1))
        return image

    def load_label(self, fname):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print('Loading lbl:%s'%fname)
        try:
            img = Image.open(fname)
        except IOError as e:
            print('Warning: no image with name %s!!'%fname)
            label = None
            return label

        label = np.array(img, dtype=np.uint8)
        label = label[np.newaxis, ...]

        return label

    def pred_to_color(self, save_path, pred):
        '''
        Input:  data_instance, should be an instance of CityDataSet.
                pred: predicted matrix, must be [1, Height, Width, 1]
        Return: colored .png image
        '''
        # Pad with RGB channels, producing [1, Height, Width, 4]
        pred = np.lib.pad(pred, ((0,0),(0,0),(0,0),(0,3)), self.padding_func)
        # Slice RGB channels
        pred = pred[:,:,:,1:4]
        H = pred.shape[1]
        W = pred.shape[2]
        pred = np.reshape(pred, (H,W,3) )

        # write to .png file
        imsave(save_path, pred)
        print('colored prediction saved to %s '%save_path)

        return pred


    def padding_func(self, vector, iaxis_pad_width, iaxis, kwargs):
        '''
        Used by
        '''
        if iaxis == 3:
            idx = vector[0]
            values = self.trainId2Color[idx]
            vector[-iaxis_pad_width[1]:] = values
        return vector



#Testing example
params = {'city_dir':"/Users/WY/Downloads/CityDatabase",
          'randomize': True,
          'seed': None}
dt = CityDataSet(params)
(img,lbl)=dt.next_batch()
print(img.shape,' ',lbl.shape)



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


# Deprecated because produced color truth image does not match the original
def color_image(image, num_classes=22):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))







