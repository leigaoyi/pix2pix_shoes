#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:42:10 2018

@author: ka
"""

import tensorflow as tf
import numpy as np
from utils import batch_data
from model.model_sngan import generator, discriminator
from utils import save_img
import time
import os

if not os.path.exists('./samples'):
    os.makedirs('./samples')
    
if not os.path.exists('./check_points/'):
    os.makedirs('./check_points/')
    
sample_dir = './samples/'
check_dir = './check_points/'

batch_size = 64
batch_idx = 29999//batch_size
train_epoch = 30
count = 0



#input node x: semantic labels  y: real figures
x_place = tf.placeholder(dtype=tf.float32, shape=[batch_size, 128, 128, 1], name='x')
y_place = tf.placeholder(dtype=tf.float32, shape=[batch_size, 128, 128, 3], name='y')
#loss and optim function


fake_gen = generator(x_place, reuse=False)

real_predict = discriminator(x_place, y_place, reuse=False)
fake_predict = discriminator(x_place, fake_gen, reuse=True)

gen_adverse = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_predict, labels=tf.ones_like(fake_predict)))
gen_l1 = tf.reduce_mean(tf.abs(y_place - fake_gen))
gen_loss = gen_adverse+gen_l1

fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_predict,labels=tf.zeros_like(fake_predict)))
real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_predict,labels=tf.ones_like(real_predict)))
dis_loss = fake_loss + real_loss
    
    
var_list = tf.trainable_variables()
g_var_list = [x for x in var_list if 'g_' in x.name]
d_var_list = [x for x in var_list if 'd_' in x.name]
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    g_opt = tf.train.AdamOptimizer(0.0002, 0.5, 0.9)
    d_opt = tf.train.AdamOptimizer(0.0002, 0.5, 0.9)
    optim_gen = g_opt.compute_gradients(gen_loss, var_list=g_var_list)
    optim_g = g_opt.apply_gradients(optim_gen)
    optim_dis = d_opt.compute_gradients(dis_loss, var_list= d_var_list)
    optim_d = d_opt.apply_gradients(optim_dis)
# the optim operation

sample_gen = generator(x_place, reuse=True)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=1)
    for epoch in range(train_epoch):
        for idx in range(batch_idx):
            start = time.time()
            figs, labels = batch_data(idx)

            feed_dict = {x_place: labels,
                         y_place: figs}
            
            sess.run(optim_d, feed_dict=feed_dict)           
      
            for i in range(1):
                sess.run(optim_g, feed_dict=feed_dict)
            
            gen_value, dis_value, l1_value = sess.run([gen_loss, dis_loss, gen_l1], feed_dict=feed_dict)
            # compute cost time
            end = time.time()
            single_time = (end-start)/60
            total_time = single_time * train_epoch * batch_idx
            remain_time = total_time - epoch*batch_idx*single_time - idx*single_time 
            
            count += 1
            print('Remain time : %.2f min , count : %d'%(remain_time, count))
            print('epoch %d, step %d '%(epoch, idx))
            print(' gen l1 loss : %.4f '%l1_value)
            print(' gen loss : %.4f \n dis loss %.4f'
                      %( gen_value, dis_value))
        
            print('')
            if (count) % 250 == 0:
                # first smaple
                dir_path = './samples/'
                fig_path = dir_path + str(count) +'.jpg'
                real_x, sample_x = batch_data(14)

                feed_dict = {x_place: sample_x}
                sample = sess.run(sample_gen, feed_dict=feed_dict)
                
#                label_1 = np.concatenate((sample_x[0,...],
#                                          sample_x[1,...],
#                                          sample_x[2,...]),
#                                            axis=1)
                sample_1 = np.concatenate((sample[0,...], sample[1,...], 
                                          sample[2,...],
                                          sample[10,...],
                                          sample[20,...]),
                                          axis=1)
                sample_fig = np.concatenate((real_x[0,...],
                                             real_x[1,...],
                                             real_x[2,...],
                                             real_x[10, ...],
                                             real_x[20, ...]), axis = 1)
                record = np.concatenate((sample_1, sample_fig),
                                        axis=0)
                save_img(record, fig_path)
                
            if count % 500 == 0:
                saver.save(sess, check_dir, count)
#    saver.save(sess, check_dir, count)
              
