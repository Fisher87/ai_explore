#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：highway.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月20日
#   描    述：highway layer.
#
#================================================================

import tensorflow as tf

def highway(inputs, output_size, 
            activation=tf.tanh, 
            scope="highway-layer",
            reuse=tf.AUTO_REUSE):
    '''
    highway layer
    @param: inputs: [batch_size, l, dim]
    '''
    input_shape = tf.shape(inputs)
    batch_size  = input_shape[0]
    seq_len = input_shape[1]
    inputs = tf.reshape(inputs, [batch_size * seq_len, output_size])
    with tf.variable_scope(scope, reuse=reuse):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, seq_len, output_size])

    return outputs

def multi_highway(inputs, output_size, num_layers, 
                  activation_func=tf.tanh, 
                  scope_name="multi-highway", 
                  reuse=tf.AUTO_REUSE):
    '''
    multi highway layer
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in range(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway(inputs, output_size, activation_func=activation_func, scope=cur_scope_name)

    return in_val
