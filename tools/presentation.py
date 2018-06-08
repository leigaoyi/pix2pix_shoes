# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:15:02 2018

@author: ky
"""

import tensorflow as tf
from PIL import Image
from utils import save_img

import glob
import os
import imageio
import numpy as np

model_name = './presentation/SNGAN.jpg'

label_path = './data/test/labels/*.jpg'
figs_path = './data/test/figs/*.jpg'
sample_path = './score/*.jpg'

sample_list = glob.glob(sample_path)
label_list = glob.glob(label_path)

def get_num(path):
    number, _ = os.path.splitext(os.path.basename(path))
    return int(number)

sample_list = sorted(sample_list, key= lambda path: get_num(path))
label_list = sorted(label_list, key=lambda path: get_num(path))

fig_gen = []
label_real = []
for i in range(5):
    img = Image.open(sample_list[i])
    img = np.array(img)
    fig_gen.append(img)
    
    label = Image.open(label_list[i])
    label = np.array(label)
    label_real.append(label)

fig = np.concatenate(fig_gen, axis=0)
label = np.concatenate(label_real, axis=0)

imageio.imwrite(model_name, fig)
imageio.imwrite('./presentation/label.png', label)



