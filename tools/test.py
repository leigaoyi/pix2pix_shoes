# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:02:52 2018

@author: ky
"""

import tensorflow as tf
import numpy as np
from utils import batch_data, test_data
from model_sngan import generator, discriminator
from utils import save_img, save_batch_imgs
import time
import os

check_dir = './check_points'
fig_dir = './score/'
batch_size = 64
batch_idx = 10000//batch_size
train_epoch = 2
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
#test files
#restore from the checkpoint dir
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
#    print('22')
    ckpt = tf.train.get_checkpoint_state(check_dir)
# restore from the check point    
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    for i in range(batch_idx):
        _, x = test_data(i)
        feed_dict = {x_place:x}
        sample_imgs = sess.run(sample_gen, feed_dict=feed_dict)
        save_batch_imgs(sample_imgs, i, fig_dir)