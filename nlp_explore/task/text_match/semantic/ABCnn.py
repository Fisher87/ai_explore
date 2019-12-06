#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ABCnn.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月06日
#   描    述：
#
#================================================================

import tensorflow as tf

class ABCnn(object):
    def __init__(self, seq_length,
                vocab_size,
                embedding_dim,
                filter_sizes,
                filter_num,
                learning_rate=0.001,
                keep_prob = 0.5,
                random_embedding=True,
                num_classes=2):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.random_embedding = random_embedding
        self.num_classes = num_classes

    def conv(self, x, namescope="conv"):
        with tf.name_scope(namescope):
            for filter_size in filter_sizes:
                filter_shape = [filter_size, self.embedding_dim, 1, self.filter_num]
                conv = tf.nn.conv2d()

    def padding(self, x):
        return tf.pad(x, paddings=[[0,0], [2,2], [0,0], [0,0]])

    def __call__(self):
        # init placeholder
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="p")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="h")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            if self.random_embedding:
                embedding_table = tf.get_variable(dtype=tf.float32, shape=[self.vocab_size, self.embedding_dim],
                                                     name="embedding_table")
            else:
                embedding_table = load_embedding()

            p_embedding = tf.nn.embedding_lookup(embedding_table, self.p)
            h_embedding = tf.nn.embedding_lookup(embedding_table, self.h)

        # convolution layer
        p_embedding_expanded = tf.expand_dims(p_embedding, axis=-1)
        h_embedding_expanded = tf.expand_dims(h_embedding, axis=-1)

        # wide convolution

