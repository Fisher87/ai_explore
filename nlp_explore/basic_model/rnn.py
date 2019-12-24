#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：rnn.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月26日
#   描    述：
#             [1]. RNN/LSTM/GRU
#             [2]. BiLSTM
#
#================================================================

import tensorflow as tf

def rnn_cell(num_units, cell_type="lstm", name="rnn_unit"):
    with tf.variable_scope(name):
        if cell_type="lstm":
            cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        else:
            cell = tf.contrib.rnn.GRUCell(num_units=dim)
        return cell


def biLSTM(inputs, dim, seq_len, name="bilstm"):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        cell_fw = rnn_cell(dim, name='fw'+name)
        cell_bw = rnn_cell(dim, name='bw'+name)
        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                           cell_bw=cell_bw, 
                                                           inputs=inputs, 
                                                           sequence_length=seq_len, 
                                                           dtype=tf.float32, 
                                                           scope=name)
        return hidden_states, cell_states



