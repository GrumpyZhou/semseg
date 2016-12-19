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


# Deprecated because produced color truth image does not match the original
def color_image(image, num_classes=22):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

'''
#Testing voc dataset
params = {'voc_dir':"../data/VOC2012",
          'dataset':'train',
          'classes':['person','cat','dog'], # All classes are loaded if class is None
          'filter_no_label':True,         # Filter all indices with no ground truth, default is False
          'randomize': True,
          'seed': None}

dt = VOCDataSet(params)
data = dt.next_batch()
'''

'''
data_config = {'city_dir':"./data/CityDatabase",
                     'randomize': True,
                     'seed': None,
                     'dataset': 'test'}
dt = CityDataSet(data_config)
(img,lbl)=dt.next_batch()
print(img.shape,' ',lbl==None)
'''
