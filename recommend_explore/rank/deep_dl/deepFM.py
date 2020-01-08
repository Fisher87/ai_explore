#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：deepFM.PY
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import numpy as np
import tensorflow as tf

class DeepFM(object):
    def __init__(self, **kwargs):
        self.feature_size = kwargs.get("feature_size")
        self.field_size = kwargs.get("field_size")
        self.embedding_size = kwargs.get("embedding_size")  # 8
        self.fm_keep_prob = kwargs.get("fm_keep_prob")
        self.deep_layer_sizes = kwargs.get("deep_layer_sizes") #[32, 32]
        self.deep_keep_prob = kwargs.get("deep_keep_prob")  # [0.5, 0.5]
        self.num_classes = kwargs.get("num_classes")
        self.learning_rate = kwargs.get("learning_rate")

    def __call__(self):
        # init placeholder
        ## feature_index (batch, field_size)
        self.feature_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="feature_inx")
        self.feature_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name="feature_value")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        
        # init weights
        # self.weights = self.init_weights()

        # embeddings 
        with tf.device("/cpu:0"):
            self.embeddings_table = tf.get_variable("feature_embedding_table", 
                                            shape=[self.feature_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer(), 
                                            dtype=tf.float32)

        # (batch, field_size, embedding_size)
        self.embeddings = tf.nn.embedding_lookup(self.embeddings_table, self.feature_index) 
        feature_value = tf.reshape(self.feature_value, [-1, self.field_size, 1])
        self.embeddings = tf.multiply(self.embeddings, feature_value)
        linear_out, interaction_output = self.fm(feature_value)
        deep_out = self.deep()

        concat_input = tf.concat([linear_out, interaction_output, deep_out], axis=1)
        logits = tf.layers.dense(concat_input,
                             self.num_classes,
                             activation=tf.nn.sigmoid)

        # loss
        self.y_prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=-1)
        y = tf.one_hot(self.y, self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss, axis=0)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)

    def fm(self, x):
        with tf.variable_scope("linear_layer"):
            w = tf.get_variable("w", shape=[self.feature_size, 1], 
                               initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b = tf.get_variable("b", shape=[1], 
                               initializer=tf.zeros_initializer())
            self.linear_output = tf.matmul(x, w) + b   #(batch, field_size, 1)
            self.linear_output = tf.dropout(self.linear_output, self.fm_keep_prob)

        with tf.variable_scope("interaction_layer"):
            # sum square part
            sum_sq_part = tf.square(tf.reduce_sum(self.embeddings, 1))  #(batch, embedding_size)
            # square sum part
            sq_sum_part = tf.reduce_sum(tf.square(self.embeddings), 1)  #(batch, embedding_size)

            self.interaction_output = 0.5 * tf.subtract(sum_sq_part, sq_sum_part)
            self.interaction_output = tf.nn.dropout(self.interaction_output, self.fm_keep_prob)

        return (self.linear_output, self.interaction_output)

    def deep(self):
        with tf.variable_scope("deep_layer"):
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size*self.embedding_size])
            for i in range(len(self.deep_layer_sizes)):
                self.y_deep = tf.layers.dense(self.y_deep, 
                                             deep_layer_sizes[i],
                                             activation=tf.nn.relu)
                self.y_deep = tf.nn.dropout(self.y_deep, self.deep_keep_prob[i])

        return self.y_deep

