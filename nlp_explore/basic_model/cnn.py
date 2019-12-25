#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：cnn.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月26日
#   描    述：
#
#================================================================

import tensorflow as tf

def nn_conv2d(inputs, filter_size, dim, out_channels=None, padding="VALID", name="conv"):
    """
    @param: filter_size: filter_height;
    @param: dim: in nlp, filter_width==dim(embedding_dim);
    @param: out_channels: output size. in nlp always set to equal dim(embedding_dim);
    tf.nn.conv2d:
        input:  A 4-D tensor.
        filters: A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels];
        strides:
        padding: 'SAME', 'VALID'
    """
    out_channels = out_channels if out_channels else dim
    with tf.name_scope(name=name):
        filters = tf.get_variable("filters", 
                                  shape=[filter_size, dim, inputs.get_shape()[2], out_channels],
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  dtype=tf.float32)
        conv = tf.nn.conv2d(inputs,
                            filters,
                            strides=[1,1,1,1], 
                            padding=padding)
        biases = tf.get_variable("biases", 
                                 shape=[dim], 
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        return tf.nn.relu(conv+biases)


def layer_conv2d(inputs, filter_size, num_filters):
    """
    tf.layers.conv2d
    """
    filter_shape = [filter_size, num_filters]
    return tf.layers.con2d(inputs, 
                           filters = num_filters,
                           kernel_size = filter_shape)

def conv1d(inputs, filter_size, out_channels, padding, name"conv1d"):
    """
    @param:filter_size: filter_width;
    @param:out_channels: 表示输出通道，可以理解为卷积核的个数
    tf.nn.conv1d:
        inputs = [batch, in_width, in_channels]是一个3-D张量
        filters = [filter_width, in_channels, out_channels]是一个3-D张量，
                         out_channels表示输出通道，可以理解为卷积核的个数
        stride = 1一个整数，步长，filters窗口移动的步长
    """
    with tf.variable_scope(name):
        filters = tf.get_variable("filters", 
                                 shape=[filter_size, inputs.get_shape()[-1], out_channels], 
                                 dtype=tf.float32)
        strides = 1
        conv = tf.nn.conv1d(inputs, filters, strides, padding)
        return conv

def max_pool(inputs, seq_length, filter_size, name="max-pooling"):
    """
    @param: inputs, 
    @param: seq_length,
    @param: filter_size,
    """
    assert seq_length==inputs.get_shape()[1]
    with tf.variable_scope(name):
        pooled = tf.nn.max_pool(inputs, 
                               ksize=[1, seq_length-filter_size+1, 1, 1],
                               strides=[1, 1, 1, 1],
                               name="pool")
        return pooled
