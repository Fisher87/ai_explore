#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：encode.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月27日
#   描    述：
#          reference: https://github.com/sjvasquez/quora-duplicate-questions/blob/master/encode.py
#
#================================================================

import tensorflow as tf


def lstm_encoder(inputs, lengths, state_size, keep_prob, scope='lstm-encoder', reuse=False):
    """
    LSTM encoder
    Args:
        inputs:  Sequence data. Tensor of shape [batch_size, max_seq_len, input_size].
        lengths: Lengths of sequences in inputs.  Tensor of shape [batch_size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.
    Returns:
        Tensor of shape [batch_size, state size] containing the final h states.
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
        return output_state.h

def multi_lstm_layer_encoder(inputs, lengths, state_size, num_layers,
                             keep_prob, scope='multi_lstm_layer_encode',
                             reuse=False):
    """
    Multi LSTM layer encoder
    """
    def single_lstm_cell(state_size, keep_prob, reuse=False):
        single_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        return single_cell
    cell = tf.contrib.rnn.MultiRNNCell(
        [single_lstm_cell(state_size, keep_prob) for _ in range(num_layers)])

    outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
    return output_state

def bidirectional_lstm_encoder(inputs, lengths, state_size, keep_prob, scope='bi-lstm-encoder', reuse=False):
    """
    Bidirectional LSTM encoder
    Args:
        inputs:  Sequence data. Tensor of shape [batch_size, max_seq_len, input_size].
        lengths: Lengths of sequences in inputs.  Tensor of shape [batch_size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.
    Returns:
        Tensor of shape [batch_size, 2*state size] containing the concatenated
        forward and backward lstm final h states.
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
        output_state = tf.concat([output_fw.h, output_bw.h], axis=1)
        return output_state


def reduce_max_encoder(inputs):
    """
    Max pooling over the time dimension
    Args:
        inputs:  Sequence data. Tensor of shape [batch_size, max_seq_len, input_size].
    Returns:
        Tensor of shape [batch_size, input_size].
    """
    return tf.reduce_max(inputs, axis=1)


def reduce_sum_encoder(inputs):
    """
    Sum pooling over the time dimension
    Args:
        inputs:  Sequence data. Tensor of shape [batch_size, max_seq_len, input_size].
    Returns:
        Tensor of shape [batch_size, input_size].
    """
    return tf.reduce_sum(inputs, axis=1)


def reduce_mean_encoder(inputs, lengths):
    """
    Max pooling over the time dimension
    Args:
        inputs:  Sequence data. Tensor of shape [batch_size, max_seq_len, input_size].
        lengths: Lengths of sequences in inputs.  Tensor of shape [batch_size].
    Returns:
        Tensor of shape [batch_size, input_size].
    """
    return tf.reduce_sum(inputs, axis=1) / tf.expand_dims(lengths, 1)


