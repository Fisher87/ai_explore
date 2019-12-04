#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ConvNet.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月04日
#   描    述：
#
#================================================================

import tensorflow as tf

class ConvNet(object):
    def __init__(self, 
                seq_length,
                vocab_size,
                embedding_dim,
                filter_sizes,
                filter_num,
                random_embedding=True):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.filter_num   = filter_num
        self.random_embedding = random_embedding

    def encoding(self, x):
        pass

    def __call__(self):
        #init placeholder
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="premise")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="hypothesis")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="drop_rate")

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            if self.random_embedding:
                embedding_table = self.get_variable(dtype=tf.float32, shape=[self.vocab_size, embedding_dim], 
                                                    name="embedding_table")
            else:
                # TODO
                embedding_table = load_embedding()

            p_embedding = tf.nn.embedding_lookup(embedding_table, self.p)
            h_embedding = tf.nn.embedding_lookup(embedding_table, self.h)

            p_embedding_expanded = tf.expand_dims(p_embedding, axis=-1)
            h_embedding_expanded = tf.expand_dims(h_embedding, axis=-1)

        # convolution + maxpool layer
        # feature extrace layer

        p_conv_output = []
        for i, filter_size in enumerate(self.filter_sizes):
            # filter_shape [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [filter_size, embedding_size, 1, filter_num]
            with tf.name_scope("feature_extract_layer-conv_maxpool-p-%s" % filter_size):
                filter_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_w")
                filter_b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name="filter_b")
                conv = tf.nn.conv2d(p_embedding_expanded, 
                                    filter_w, 
                                    strides=[1, 1, 1, 1],
                                    padding="VALID")
                conv_ = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # k-max pooling instead of a single max value.
                # k-max pooling means select more important chars/words, conv_[batch_size, seq_length, 1, filter_num],
                # so need to transpose dim_2 to axis -1 ==> (transpose(conv_, [0,3,2,1]));
                #   tf.nn.top_k() return axis=-1 top k `values`` and `indices`
                p_top_k_maxpooling = tf.reshape(tf.nn.top_k(tf.transpose(conv_, [0,3,2,1]), k=self.k), 
                                                  [-1, self.k*self.filter_num])
                p_conv_output.append(p_top_k_maxpooling)



    


