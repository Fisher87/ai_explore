#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ESIM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#
#================================================================

import tensorflow as tf

class ESIM(object):
    def __init__(self, 
                 seq_length, 
                 num_classes,
                 embedding_dim,
                 vocab_size, 
                 hidden_size,
                 learning_rate=0.001,
                 random_embedding=True):
        self.hidden_size = 

        self.p = tf.placeholder(tf.int32, [None, seq_length], name="premise")
        self.h = tf.placeholder(tf.int32, [None, seq_length], name="hypothesis")
        self.y = tf.placeholder(tf.int32, [None], name="y")

        # real length
        self.p_len = tf.placeholder(tf.int32, [None], name="p_len")
        self.h_len = tf.placeholder(tf.int32, [None], name="h_len")

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            if random_embedding:
                self.embedding_table = tf.get_variable(name="embedding", dtype=tf.float32, shape=[vocab_size, embedding_dim])
            else:
                # TODO
                self.embedding_table = load_embedding()

            self.embedding_p = tf.nn.embedding_lookup(self.embedding_table, self.p)
            self.embedding_h = tf.nn.embedding_lookup(self.embedding_table, self.h)

        # input encoding, `BiLSTM`
        (p_f, p_b), _ = encoding(self.embedding_p, self.p_len, reuse=tf.AUTO_REUSE, scope_name="lstm_p")
        (h_f, h_b), _ = encoding(self.embedding_h, self.h_len, reuse=tf.AUTO_REUSE, scope_name="lstm_h")

        p = self.dropout(tf.concat([p_f, p_b], axis=2))  # [batch_size, max_time, hidden_size*2]
        h = self.dropout(tf.concat([h_f, h_b], axis=2))

        # local inference modeling layer
        with tf.name_scope("local_inference_modeling"):
            # e_ij = a_i.T * b_j
            pass

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def encoding(self, x, seq_len=None, reuse=False, scope_name="encoding"):
        with tf.variable_scope(scope_name, reuse=reuse)
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

            return tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, 
                                                   bw_lstm_cell, 
                                                   x, 
                                                   sequence_length=seq_len,
                                                   dtype=tf.float32)

