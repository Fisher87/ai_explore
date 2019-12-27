#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：layer_utils.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月27日
#   描    述：
#
#================================================================

import tensorflow as tf

from general import shape

def lstm_layer(inputs, lengths, state_size, keep_prob, scope='lstm-layer', reuse=False):
    """
    LSTM layer.
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.
    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        return outputs


def bidirectional_lstm_layer(inputs, lengths, state_size, keep_prob, scope='bi-lstm-layer', reuse=False):
    """
    Bidirectional LSTM layer.
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.
    Returns:
        Tensor of shape [batch size, max sequence length, 2*state_size] containing the concatenated
        forward and backward lstm outputs at each timestep.
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        cell_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        outputs = tf.concat(outputs, 2)
        return outputs


def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, dropout=None,
                                 scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return z


def temporal_convolution_layer(inputs, output_units, convolution_width, bias=True, activation=None,
                               dropout=None, scope='time-distributed-conv-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.
    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps (words) to use in convolution.
    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, W, padding='SAME', strides=[1])
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return z


def dense_layer(inputs, output_units, bias=True, activation=None, dropout=None, scope='dense-layer',
                reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
    Returns:
        Tensor of shape [batch size, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return z

