# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:50:33 2018

@author: kasy
"""

#import matplotlib.pyplot as plt
import imageio
import os
import glob

images = []
dir_path = './cloud_sample/*.jpg'
listdir = glob.glob(dir_path)

def get_number(path):
    number, _ = os.path.splitext(os.path.basename(path))
    return int(number)

filenames = sorted(listdir, key=lambda path: get_number(path))

#filenames=sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('gif.gif', images,duration=1)