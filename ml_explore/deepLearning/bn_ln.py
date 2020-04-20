#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：bn_ln.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月20日
#   描    述：BatchNorm & LayerNorm 实现
#
#================================================================

import tensorflow as tf
import numpy as np

# a = np.array([float(i) for i in range(60)]).reshape([3,4,5])
a = np.random.rand(5,3,10)    #(B, L, D)
x = tf.convert_to_tensor(a)

bn = tf.contrib.layers.batch_norm(x, scope="bn")
ln = tf.contrib.layers.layer_norm(x, scope="ln")

def batch_norm(x, eps=1e-5):
    '''batch normalization: 对每个神经元的batch size进行操作;
    '''
    mean, var = tf.nn.moments(x, [0, 1], keep_dims=True)
    x_normalized = (x-mean) / tf.math.sqrt(var + eps)
    gamma = 1.0
    beta = 0.0
    return gamma * x_normalized + beta

def layer_norm(x, eps=1e-5):
    '''layer normalization: 对每层的所有神经元进行操作;
    '''
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_normalized = (x-mean) / tf.math.sqrt(var + eps)
    gamma = 1.0
    beta = 0.0
    return gamma * x_normalized + beta

_batch_norm = batch_norm(x)
_layer_norm = layer_norm(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _x, b, l, _b, _l = sess.run([x, bn, ln, _batch_norm, _layer_norm])
    print(_x)
    print("*****"*20)
    print(b)
    print("*****"*20)
    print(_b)
    print("*****"*20)
    print(l)
    print("*****"*20)
    print(_l)


