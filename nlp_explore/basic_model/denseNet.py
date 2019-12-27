#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：denseNet.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月27日
#   描    述：
#          reference: https://zhuanlan.zhihu.com/p/37189203
#
#================================================================

import tensorflow as tf

dense_net_first_scale_down_ratio = 0.3
dense_growth_rate = 10
dense_net_transition_rate = 0.5

def dense_net(v):
    filters = int(v.shape[-1] * dense_net_first_scale_down_ratio)
    v_in = tf.layers.conv2d(v, filters=filters, kernel_size=(1, 1))
    for _ in range(3):
        for _ in range(8):
            v_out = tf.layers.conv2d(v_in,
                                     filters=dense_growth_rate,
                                     kernel_size=(3, 3),
                                     padding='SAME',
                                     activation='relu')
            v_in = tf.concat((v_in, v_out), axis=-1)
        transition = tf.layers.conv2d(v_in,
                                      filters=int(v_in.shape[-1].value * dense_net_transition_rate),
                                      kernel_size=(1, 1))
        transition_out = tf.layers.max_pooling2d(transition,
                                                 pool_size=(2, 2),
                                                 strides=2)
        v_in = transition_out
    return v_in
