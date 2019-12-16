#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：BIMPM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月12日
#   描    述：
#
#================================================================

import tensorflow as tf

class Bimpm(object):
    def __init__(self, seq_len, word_embedding_dim, char_embedding_dim, char_vocab_len
                 char_hidden_size, num_perspective):
        self.seq_len = seq_len
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_vocab_len = char_vocab_len
        self.char_hidden_size = char_hidden_size
        self.num_perspective= num_perspective

    def lstm(self, x, scope="lstm"):
        with tf.variable_scope(scope, reuse=None):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden_size)
            return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def dropout(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def __call__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name="p")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name="h")
        self.p_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.word_embedding_dim], 
                                    name="p_word")
        self.h_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.word_embedding_dim], 
                                    name="h_word")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")
        self.embed = tf.get_variable(dtype=tf.float32, shape=[self.char_vocab_len, self.char_embedding_dim],
                                    name="char_embed_table")
        for i in range(1,9):
            setattr(self, f'w{i}', 
                    tf.get_variable(dtype=tf.float32, shape=[self.num_perspective, self.char_hidden_size],
                                    name=f'w{i}')
                   )

        # input layer
        with tf.device("/cpu:0"), tf.name_scope("char_embedding"):
            p_char_embedding = tf.nn.embedding_lookup(self.embed, self.p)
            h_char_embedding = tf.nn.embedding_lookup(self.embed, self.h)

        p_output, _ = self.lstm(p_char_embedding, scope="lstm_p")
        h_output, _ = self.lstm(h_char_embedding, scope="lstm_h")
        ## concat char_embedding & word_embedding
        p_embedding = tf.concat([self.p_vec, p_output], axis=-1)
        h_embedding = tf.concat([self.h_vec, h_output], axis=-1)
        p_embedding = self.dropout(p_embedding)
        h_embedding = self.dropout(h_embedding)

        # context representation layer

