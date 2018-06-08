# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:03:03 2018

@author: ky
"""

import tensorflow as tf
import sys
sys.path.append('..')
from ops import lrelu, bn, sn

batch_size = 64

def generator(x_in, reuse=False):
    ngf = 64
    conv_list = [[5, 5, 1, ngf], 
                 [5, 5, ngf, ngf*2], 
                 [5, 5, ngf*2, ngf*4], 
                 [5, 5, ngf*4, ngf*8],
                 [5, 5, ngf*8, ngf*8],
                 [5, 5, ngf*8, ngf*8],
                 [5, 5, ngf*8, ngf*8],
                 [5, 5, ngf*8, ngf*8]]
    res_list = [[5, 5, ngf*8, ngf*8]]
#    deconv_list = [[5, 5, ngf*8*2, ngf*8], 
#                   [5, 5, ngf*8*2, ngf*8*2],
#                   [5, 5, ngf*8*2, ngf*8*2],
#                   [5, 5, ngf*8*2, ngf*8*2],
#                   [5, 5, ngf*4*2, ngf*8*2],
#                   [5, 5, ngf*2*2, ngf*4*2],
#                   [5, 5, ngf*2, ngf*2*2]]
#    out_shape = [[1, 2, 2, ngf*8*2], 
#                 [1, 4, 4, ngf*8*2],
#                 [1, 8, 8, ngf*8*2],
#                 [1, 16, 16, ngf*8*2],
#                 [1, 32, 32, ngf*4*2],
#                 [1, 64, 64, ngf*2*2],
#                 [1, 128,128, ngf*2],
#                 [1, 256, 256, 3]]
    deconv_list = [[5, 5, ngf*4*2, ngf*8],
                   [5, 5, ngf*2*2, ngf*4*2],
                   [5, 5, ngf*2, ngf*2*2]]
    out_shape = [[64, 16, 16, ngf*4*2],
                 [64, 32, 32, ngf*2*2],
                 [64, 64,64, ngf*2],
                 [64, 128, 128, 3]]
    layers = []
    layers.append(x_in)
    for i in range(4):
        with tf.variable_scope('g_conv_%d'%i, reuse=reuse) as scope:
            k = tf.get_variable(scope.name+'_k', shape=conv_list[i],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0.0,0.02))
            pre = tf.nn.conv2d(layers[-1], k, [1, 2, 2, 1], 'SAME')
            pre = tf.nn.relu(bn(pre))
            layers.append(pre)
        
    for i in range(2):
        with tf.variable_scope('g_resblock_%d'%i, reuse=reuse) as scope:
            k = tf.get_variable(scope.name+'_k', shape=res_list[0],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0.0, 0.02))
            pre = tf.nn.conv2d(layers[-1], k, [1, 1, 1, 1], 'SAME')
            pre = tf.nn.relu(bn(pre))
            layers.append(pre)
            
    for i in range(3):
        with tf.variable_scope('g_deconv_%d'%i, reuse=reuse) as scope:
            k = tf.get_variable(scope.name+'_k', shape=deconv_list[i],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0.0, 0.02))
            pre = tf.nn.conv2d_transpose(layers[-1], k, out_shape[i], [1, 2, 2, 1], 'SAME')
            pre = tf.nn.relu(bn(pre))
            if i < 2:
                pre = tf.nn.dropout(pre, keep_prob=0.5)
            layers.append(pre)
            
    with tf.variable_scope('g_out', reuse=reuse) as scope:
        k =tf.get_variable(scope.name+'_k', [5, 5, 3, ngf*2],
                           dtype=tf.float32,
                           initializer=tf.random_normal_initializer(0.0, 0.02))
        out = tf.nn.conv2d_transpose(layers[-1], k, [batch_size, 128, 128, 3], [1, 2, 2, 1], 'SAME')
        layers.append(out)
    gen_out = layers[-1]
    gen_out = tf.nn.tanh(gen_out)        
    return gen_out

def discriminator(x_in, y_in, reuse=False):
    ndf = 64
    input_ = tf.concat((y_in, x_in), axis=3)    
#
    
    with tf.variable_scope('d_conv_1', reuse=reuse) as scope:
        k = tf.get_variable(scope.name+'_k', [5, 5, 4, ndf],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0.0, 0.02))
        k = sn(k)
        pre = tf.nn.conv2d(input_, k, [1, 2, 2, 1], 'SAME')
        pre = lrelu(pre)
    
    with tf.variable_scope('d_conv_2', reuse=reuse) as scope:
        k = tf.get_variable(scope.name+'_k', [5, 5, ndf, ndf*2],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
        k = sn(k)
        conv2 = tf.nn.conv2d(pre, k, [1, 2, 2, 1], 'SAME')
        conv2 = lrelu(conv2)
        
    with tf.variable_scope('d_conv_3',reuse=reuse) as scope:
        k = tf.get_variable(scope.name+'_k', [5, 5, ndf*2, ndf*4],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
        k = sn(k)
        conv3 = tf.nn.conv2d(conv2, k, [1, 2, 2, 1],'SAME')
        conv3 = lrelu(conv3)
    layers = []
    layers.append(conv3)    
    for i in range(1):
        with tf.variable_scope('d_rescov_%d'%i, reuse=reuse) as scope:
            k = tf.get_variable(scope.name+'_k', [5, 5, ndf*4, ndf*8],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0., 0.02))
            k = sn(k)
            res = tf.nn.conv2d(layers[-1], k, [1, 1, 1, 1], 'SAME')
            res = lrelu(res)
            layers.append(res)
    with tf.variable_scope('d_out', reuse=reuse) as scope:
        k = tf.get_variable(scope.name+'_k', [5, 5, ndf*8, 1],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02))
        
        out = tf.nn.conv2d(layers[-1], k, [1, 1, 1, 1], 'SAME')
        out = tf.sigmoid(out)# pix2pix
        layers.append(out)
    
        
    return out