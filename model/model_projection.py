# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:16:45 2018

@author: ky
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:35:11 2018

@author: ka
"""

import tensorflow as tf
import sys
sys.path.append('..')
from ops import lrelu
from ops import bn
from ops import sn

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

    deconv_list = [[5, 5, ngf*4*2, ngf*8],
                   [5, 5, ngf*2*2, ngf*4*2],
                   [5, 5, ngf*2, ngf*2*2]]
    
    deconv_concate = [[5, 5, ngf*2, ngf*6], [5, 5, 3, ngf*3]]
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
            
    for i in range(2):
        with tf.variable_scope('g_deconv_%d'%i, reuse=reuse) as scope:
            k = tf.get_variable(scope.name+'_k', shape=deconv_list[i],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0.0, 0.02))
            pre = tf.nn.conv2d_transpose(layers[-1], k, out_shape[i], [1, 2, 2, 1], 'SAME')
            pre = tf.nn.relu(bn(pre))
            pre = tf.nn.dropout(pre, keep_prob=0.5)
            layers.append(pre)
    with tf.variable_scope('g_deconv_3', reuse=reuse) as scope:
        input_ = tf.concat((layers[-1], layers[2]), axis=3)
        k = tf.get_variable(scope.name+'_k', shape= deconv_concate[0],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0.0, 0.02))
        pre = tf.nn.conv2d_transpose(input_, k, output_shape=out_shape[2],
                                     strides=[1, 2, 2, 1], padding='SAME')
        pre = tf.nn.relu(pre)
        layers.append(pre)
            
    with tf.variable_scope('g_out', reuse=reuse) as scope:
        # u-skip to the last layer
        input_ = tf.concat((layers[-1], layers[1]), axis=3)
        k =tf.get_variable(scope.name+'_k', [5, 5, 3, ngf*3],
                           dtype=tf.float32,
                           initializer=tf.random_normal_initializer(0.0, 0.02))
        out = tf.nn.conv2d_transpose(input_, k, [batch_size, 128, 128, 3], [1, 2, 2, 1], 'SAME')
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
    
    with tf.variable_scope('d_projection', reuse=reuse) as scope:
        V = tf.get_variable(scope.name+'_w', [128*128, batch_size],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 0.02))
        x = tf.reshape(x_in, (batch_size, -1))
        y = tf.reshape(layers[-1], (batch_size, -1))
        pre = tf.matmul(x,V)
        out = tf.matmul(pre, y)
        
    return out