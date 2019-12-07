#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ABCnn.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月06日
#   描    述：
#       reference: [1]. https://arxiv.org/pdf/1512.05193.pdf
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
                num_classes=2,
                abcnn_1=False):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.random_embedding = random_embedding
        self.num_classes = num_classes

    def _conv(self, x, namescope="conv"):
        kernel = tf.get_variable("kernel", shape=[self.filter_size, self.embedding_dim, x.get_shape()[2], self.embedding_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        conv = tf.nn.conv2d(x, kernel, 
                           strides=[1,1,1,1], padding="VALID")
        biases = tf.get_variable("biases", shape=[self.embedding_dim], dtype=tf.float32, 
                                initializer=tf.constant_initializer(0.0))
        
        return tf.nn.relu(conv+biases)

    def padding(self, x, w):
        return tf.pad(x, paddings=[[0,0], [2,2], [0,0], [0,0]])

    def __call__(self):
        # init placeholder
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="p")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="h")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        self.W0 = tf.get_variable(name="W0",
                                 shape=[self.seq_length+4, self.embedding_dim],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004))

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            if self.random_embedding:
                embedding_table = tf.get_variable(dtype=tf.float32, shape=[self.vocab_size, self.embedding_dim],
                                                     name="embedding_table")
            else:
                embedding_table = load_embedding()

            p_embedding = tf.nn.embedding_lookup(embedding_table, self.p)
            h_embedding = tf.nn.embedding_lookup(embedding_table, self.h)

        p_embedding_expanded = tf.expand_dims(p_embedding, axis=-1)
        h_embedding_expanded = tf.expand_dims(h_embedding, axis=-1)

        # wide convolution
        p_embedding = self.padding(p_embedding_expanded)
        h_embedding = self.padding(h_embedding_expanded)

        # abcnn-1&abcnn-3 add attention layer for input layer 
        if self.abcnn_1:
            # get attention matrix
            # attention matrix: score(x,y) = 1 / (1 + |x-y|)
            educlidean = tf.sqrt(
                tf.reduce_sum(
                    tf.square(tf.transpose(p_embedding, perm=[0,2,1,3])-tf.transpose(h_embedding, perm=[0,2,3,1])),
                    axis = 1) + 1e-6)
            attention_matrix = 1 / (educlidean + 1)  # [batchsize, lp, lh]

            # p_attention = tf.matmul(
            #     attention_matrix, 
            #     tf.tile(tf.expand_dims(W0, axis=0), [attention_matrix.shape[0], 1, 1])
            # )
            # h_attention = tf.matmul(
            #     tf.transpose(attention_matrix,perm=[0,2,1]) 
            #     tf.tile(tf.expand_dims(W0, axis=0), [attention_matrix.shape[0], 1, 1])
            # )
            # p_attention = tf.expand_dims(p_attention, axis=-1)
            # h_attention = tf.expand_dims(h_attention, axis=-1)

            # 跟上面的效果相同, use einsum function
            p_attention = tf.expand_dims(tf.einsum("ijk,kl->ijl", attention_matrix, self.W0), axis=-1)   # [batchsize, lp, dim, 1]
            h_attention = tf.expand_dims(tf.einsum("ijk,kl->ijl", 
                                       tf.transpose(attention_matrix, perm=[0,2,1]), self.W0), axis=-1)  # [batchsize, lh, dim, 1]

            p_embedding = tf.concat([p_embedding, p_attention], axis=-1)
            h_embedding = tf.concat([h_embedding, h_attention], axis=-1)

        # convolution layer-1
        with tf.name_scope("conv-layer-1"):
            filter_shape = [self.filter_size, self.embedding_dim]
            p = tf.layers.conv2d(p_embedding, 
                                 filters=self.cnn1_filters,
                                 kernel_size =filter_shape)
            h = tf.layers.conv2d(h_embedding, 
                                 filters=self.cnn1_filters,
                                 kernel_size =filter_shape)
            # p = self._conv(p_embedding)
            # h = self._conv(h_embedding)
            p = self.dropout(p)
            h = self.dropout(h)

        if self.abcnn2:
            pass
        else:
            p = tf.reshape(p, shape=[-1, p.shape[1], p.shape[2]*p.shape[3]])
            h = tf.reshape(h, shape=[-1, h.shape[1], h.shape[2]*h.shape[3]])

        p = tf.expand_dims(p, axis=-1)
        h = tf.expand_dims(h, axis=-1)

        # max pooling
        p_all = 

