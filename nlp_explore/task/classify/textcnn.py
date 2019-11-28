#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：textcnn.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月28日
#   描    述：
#
#================================================================

import tensorflow as tf
# from embedding_utils import load_embedding

class TextCNN(object):
    def __init__(self, sequence_length, 
                 num_classes, 
                 vocab_size,
                 embedding_size,
                 learning_rate, 
                 filter_sizes,
                 random_embedding = True,
                 l2_lambda = 0.0005
                ):
        """

        """
        # placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if random_embedding:
                self.embedding_table = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
                                     name="W")
                self.embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_x)

            else:
                # TODO
                self.embedding_table = load_embedding()
                self.embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_x)

        # create conv and maxpool layer for each filter
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-layer-%s" %filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.truncated_normal([num_filters]), name="b")
                conv = tf.nn.relu(tf.nn.conv2d(self.embedding, W, strides=[1,1,1,1], padding="VALID")+b, name="conv")

                # max pool 
                pooled = tf.nn.max_pool(
                    conv, ksize=[1, sequence_length-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs.append(pooled)

        # concat all pooled features
        num_filters_total = num_filters * len(filter_size)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout 
        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # fnn, score and prediction
        with tf.name_scope("output"):
            w = tf.get_variable("w", shape=[num_filters_total, num_classes], 
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.truncated_normal([num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(self.h_dropout, w) + b # self.scores = tf.nn.xw_plus_b(self.h_dropout, w, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="prediction")

        # loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss

        # accuracy
        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")
