# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:59:04 2018

@author: kasy
"""

import tensorflow as tf

def bn(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, 
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, leaky=0.1):
    return tf.maximum(x, leaky*x)

def sn(inputs):
    sigma = max_singular_value(inputs, ip=1)
    w = inputs/sigma
    return w



def max_singular_value(w_mat, ip=1):
    W = tf.reshape(w_mat, (w_mat.shape[0].value, -1))
    u = tf.random_normal(shape=(1, W.shape[0].value), dtype=tf.float32)
    _u = u
    for _ in range(ip):
        a_u = tf.matmul(_u, W)
        norm_u = tf.norm(a_u, ord='euclidean')
        _v = a_u / (norm_u+1e-12)
        _v = tf.transpose(_v)
        a_v = tf.matmul(W, _v)
        norm_v = tf.norm(a_v, ord='euclidean')
        _u = a_v / (norm_v+1e-12)
        _u = tf.transpose(_u)
    sigma = tf.matmul(_u,tf.matmul(W,_v))
#    sigma = np.sum(sigma)
    return sigma