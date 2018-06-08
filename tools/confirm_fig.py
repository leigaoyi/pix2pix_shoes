# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:25:44 2018

@author: ky
"""

import glob
import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import imageio
from PIL import Image

batch_size =64

# where the dataset place
#img_dir = './data/train/labels/*.jpg'
#ori_dir = './data/train/figs/*.jpg'

img_dir = './data/train_shoes/labels/*.jpg'
ori_dir = './data/train_shoes/figs/*.jpg'

if not os.path.exists('./data/train/labels/'):
    print("The dataset files is not in the right place!")

sl_list = glob.glob(img_dir) # semantic label
rf_list = glob.glob(ori_dir) # real figure


def get_number(path):
    number, _ = os.path.splitext(os.path.basename(path))
    return int(number)

sl_list = sorted(sl_list, key=lambda path : get_number(path))
rf_list = sorted(rf_list, key=lambda path : get_number(path))

for i in range(len(sl_list)):
    img = Image.open(sl_list[i])
    img = np.array(img)
    if img.shape[0] != 128 :
        os.remove(sl_list[i])
        os.remove(rf_list[i])
    if i% 1000 == 0:
        print('Runing')