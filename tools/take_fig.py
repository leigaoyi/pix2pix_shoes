# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:04:12 2018

@author: kasy
"""
import glob
from PIL import Image
import numpy as np
import imageio
import os

dir_path = './cloud_sample/*.jpg'
img_list = glob.glob(dir_path)

for i in img_list:
    img = Image.open(i)
    img = np.array(img,dtype=np.float32)
    fig = img[:128, :128, :]
    file_name = os.path.basename(i)
    fig_path = './score/'+file_name
    imageio.imwrite(fig_path, fig)
    