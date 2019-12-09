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
                filter_num,
                filter_size=3,
                learning_rate=0.001,
                keep_prob = 0.5,
                cnn1_filters = 256,
                cnn2_filters = 128,
                random_embedding=True,
                num_classes=2,
                abcnn_1 = False,
                abcnn_2 = False,
                ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.cnn1_filters = cnn1_filters
        self.cnn2_filters = cnn2_filters
        self.random_embedding = random_embedding
        self.num_classes = num_classes
        self.abcnn_1 = abcnn_1
        self.abcnn_2 = abcnn_2

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
                                 kernel_size =filter_shape)        # [batchsize, lp+1-filter_width, 1, cnn1_filters]
            h = tf.layers.conv2d(h_embedding, 
                                 filters=self.cnn1_filters,
                                 kernel_size =filter_shape)        # [batchsize, lh+1-filter_width, 1, cnn1_filters]
            # p = self._conv(p_embedding)
            # h = self._conv(h_embedding)
            p = self.dropout(p)
            h = self.dropout(h)

        # convolution layer-1
        # maxpooling layer
        # 原论文中有两层宽卷积操作

        # abcnn-2&abcnn-3 add attention layer for conv layer
        if self.abcnn_2:
            attention_pool_edulidean = tf.sqrt(
                tf.reduce_mean(
                    tf.square(tf.transpose(p, perm=[0,3,1,2]) - tf.transpose(h, perm=[0,3,2,1])),
                    , axis=-1))
            attention_pool_matrix = 1 / (attention_pool_edulidean + 1)   # [batchsize, lp+1-filter_width, lh-1+filter_width]

            # col-wise sum
            p_sum = tf.reduce_sum(attention_pool_matrix, axis=2, keep_dims=True)         # [batchsize, lp+1-filter_width, 1]
            # row-wise sum
            h_sum = tf.reduce_sum(attention_pool_matrix, axis=1, keep_dims=True)         # [batchsize, 1, lh+1-filter_width]

            p = tf.reshape(p, shape=[-1, p.shape[1], p.shape[2]*p.shape[3]])        # [batchsize, lp+1-filter_width, cnn1_filters] 
            h = tf.reshape(h, shape=[-1, h.shape[1], h.shape[2]*h.shape[3]])        # [batchsize, lh+1-filter_width, cnn1_filters]

            p = tf.multiply(p, p_sum)   # [batchsize, lp+1-filter_width, cnn1_filters]
            h = tf.multiply(h, h_sum)   # [batchsize, lh+1-filter_width, cnn1_filters]
        else:
            p = tf.reshape(p, shape=[-1, p.shape[1], p.shape[2]*p.shape[3]])   # [batchsize, lp+1-filter_width, cnn1_filters] 
            h = tf.reshape(h, shape=[-1, h.shape[1], h.shape[2]*h.shape[3]])   # [batchsize, lh+1-filter_width, cnn1_filters]

        p = tf.expand_dims(p, axis=-1)
        h = tf.expand_dims(h, axis=-1)

        with tf.name_scope("conv-layer2"):
            filter_shape = [self.filter_size, self.cnn1_filters]
            p = tf.layers.conv2d(p, 
                                 filters=self.cnn2_filters,
                                 kernel_size = filter_shape)
            h = tf.layers.conv2d(h, 
                                 filters=self.cnn2_filters,
                                 kernel_size = filter_shape)
            p = self.dropout(p) 
            h = self.dropout(h)

        # max pooling all, through word direction[dim=1].
        p_all = tf.reduce_mean(p, axis=1)   # [batchsize, 1, cnn2_filters] 
        h_all = tf.reduce_mean(h, axis=1)   # [batchsize, 1, cnn2_filters]

        all_embedding = tf.concat([p_all, h_all], axis=-1)
        shape = all_embedding.shape
        all_embedding = tf.reshape(all_embedding, [-1, shape[1]*shape[2]])

        # fc layer
        out = tf.layers.dense(all_embedding, 50)
        self.logits = tf.layers.dense(out, self.num_classes)

        # train opt
        with tf.name_scope("train-opt"):
            y = tf.one_hot(self.y, self.num_classes)
            loss = tf.nn.softmax_entropy_with_logits(logits=self.logits, labels=y)
            self.loss = tf.reduce_mean(loss, axis=0)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.predictions = tf.argmax(self.logits, axis=-1)
            self.correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
            self.acc = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32), axis=0)

