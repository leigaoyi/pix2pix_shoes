# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:37:45 2018

@author: kasy
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
#img_dir = './data/labels/*.jpg'
#ori_dir = './data/figs/*.jpg'

img_dir = './data/train_shoes/labels/*.jpg'
ori_dir = './data/train_shoes/figs/*.jpg'
test_dir = './data/test/labels/*.jpg'

if not os.path.exists('./data/train/labels/'):
    print("The dataset files is not in the right place!")

sl_list = glob.glob(img_dir) # semantic label
rf_list = glob.glob(ori_dir) # real figure
test_list = glob.glob(test_dir)


def get_number(path):
    number, _ = os.path.splitext(os.path.basename(path))
    return int(number)

sl_list = sorted(sl_list, key=lambda path : get_number(path))
rf_list = sorted(rf_list, key=lambda path : get_number(path))
test_list = sorted(test_list, key=lambda path : get_number(path))

def batch_data(index): # read batch semantic labels

    with tf.name_scope("load_batch_data"):
        ori_img = []
        label_img = []
        for i in range(batch_size):
            label_index = sl_list[index*batch_size+i]
            fig_index = rf_list[index*batch_size+i]
            
            label = Image.open(label_index)
            fig = Image.open(fig_index)
            # [0, 255] --> [-1. 1]
            label = (np.array(label)/255 - 0.5)*2
            fig = (np.array(fig)/255 - 0.5)*2
            #some data augmentation : add noise
            label_noise = np.random.normal(0.0, 5.0, size=label.shape)
            fig_noise = np.random.normal(0.0, 5.0, size=fig.shape)
            
            label = label + 0.001 * label_noise
            fig = fig + 0.001 * fig_noise
            # restrict to [-1, 1]
            label = np.minimum(label, 1.0)
            fig = np.minimum(fig, 1.0)
            
            label = np.maximum(label, -1.0)
            fig = np.maximum(fig, -1.0)
            
            ori_img.append(fig)
            label_img.append(label)
    ori_img = np.reshape(ori_img,(batch_size, 128, 128, 3))
    label_img = np.reshape(label_img, (batch_size, 128, 128 ,1))            
    
    return ori_img, label_img

def test_data(index):
    with tf.name_scope("load_batch_data"):
        ori_img = []
        label_img = []
        for i in range(batch_size):
            label_index = test_list[index*batch_size+i]
            fig_index = rf_list[index*batch_size+i]
            
            label = Image.open(label_index)
            fig = Image.open(fig_index)
            # [0, 255] --> [-1. 1]
            label = (np.array(label)/255 - 0.5)*2
            fig = (np.array(fig)/255 - 0.5)*2
            #some data augmentation : add noise
            label_noise = np.random.normal(0.0, 5.0, size=label.shape)
            fig_noise = np.random.normal(0.0, 5.0, size=fig.shape)
            
            label = label + 0.001 * label_noise
            fig = fig + 0.001 * fig_noise
            # restrict to [-1, 1]
            label = np.minimum(label, 1.0)
            fig = np.minimum(fig, 1.0)
            
            label = np.maximum(label, -1.0)
            fig = np.maximum(fig, -1.0)
            
            ori_img.append(fig)
            label_img.append(label)
    ori_img = np.reshape(ori_img,(batch_size, 128, 128, 3))
    label_img = np.reshape(label_img, (batch_size, 128, 128 ,1))            
    
    return ori_img, label_img

def save_img(fig, path):
#    _, height, width, _ = fig.shape
    im = (fig+1)/2 * 255.99
    im = np.array(im, dtype=np.uint8)
    
    imageio.imwrite(path, im)
    return 0

def save_batch_imgs(figs, idx, dir_path):
    for i in range(64):
        fig_path = dir_path + str(64*idx + i) +'.jpg'
        save_img(figs[i], fig_path)
    print("Finish index %d"%idx)