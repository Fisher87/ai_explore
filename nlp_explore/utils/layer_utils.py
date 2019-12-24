#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：layer_utils.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月20日
#   描    述：
#
#================================================================

import tensorflow as tf

def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask

def dropout_layer(inputs, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


def cosine_distance_array(v1, v2, cosine_norm=True, eps=1e-6):
    """
    cosine for array/vector.
    @param v1: 2d-tensor, (dim, 1), e.g.([....., a, 1, d])
    @param v2: 2d-tensor, (dim, 1), e.g.([....., b, a, 1])
    @parma cosine_norm: boolean, wether to normalize.
    """
    cosine_numerator = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1), eps))
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1), eps))

    return cosine_numerator / v1_norm / v2_norm

def cosine_distance_matrix(m1, m2, eps=1e-6):
    '''
    cosine for two matrix.
    @param m1: 3d-tensor, (batch, l1, dim).
    @param m2: 3d-tensor, (batch, l2, dim).
    '''
    cosine_numerator = tf.matmul(m1, tf.transpose(l2, perm=[0, 2, 1]))
    norm = tf.norm(m1, axis=-1, keep_dims=True) * tf.transpose(
                               tf.norm(m2, axis=-1, keep_dims=True), perm=[0,2,1])
    cosine = tf.divide(cosine_numerator, norm)

    return cosine
    
def euclidean_distance(v1, v2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance

