#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：DCN.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：deep&cross
#
#================================================================

import tensorflow as tf


class DeepCross(object):
    def __init__(self, **kwargs):
        self.feature_size = kwargs.get("feature_size")
        self.field_size = kwargs.get("field_size")
        self.continue_x_size = kwargs.get("continue_x_size")
        self.embedding_size = kwargs.get("embedding_size")  # 8
        self.deep_layer_sizes = kwargs.get("deep_layer_sizes") #[32, 32]
        self.deep_keep_prob = kwargs.get("deep_keep_prob")  # [0.5, 0.5]
        self.cross_layer_num = kwargs.get("cross_layer_num")
        self.cross_keep_probs = kwargs.get("cross_keep_probs")
        self.num_classes = kwargs.get("num_classes")
        self.learning_rate = kwargs.get("learning_rate")
        self.total_size = self.field_size * self.embedding_size + self.continue_x_size

    def init_weights(self):
        for i in range(self.cross_layer_num):
            w = tf.get_variable("cross_w%s" %i, shape=[self.total_size, 1], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.Variable("cross_b%s" %i, shape=[self.total_size], 
                            initializer=tf.zeros_initializer(), dtype=tf.float32)
            setattr(self, "cross_w%s" %i, w)
            setattr(self, "cross_b%s" %i, b)
                               )

    def __call__(self):
        # init placeholder
        self.category_x = tf.placeholder(tf.float32, shape=[None, self.category_feature_size], name="category_x")
        self.category_inx = tf.placeholder(tf.float32, shape=[None, self.category_field_size], name="category_inx")
        self.continue_x = tf.placeholder(tf.float32, shape=[None, self.continue_feature_size], name="continue_x")
        self.continue_inx = tf.placeholder(tf.float32, shape=[None, self.continue_field_size], name="continue_inx")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        
        # init weights
        self.weights = self.init_weights()

        # embeddings 
        with tf.device("/cpu:0"):
            self.embeddings_table = tf.get_variable("feature_embedding_table", 
                                            shape=[self.feature_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer(), 
                                            dtype=tf.float32)

        # (batch, field_size, embedding_size)
        self.embeddings = tf.nn.embedding_lookup(self.embeddings_table, self.category_inx) 
        category_x = tf.reshape(self.category_x, [-1, self.field_size, 1])
        self.embeddings = tf.multiply(self.embeddings, self.category_x)

        self.x0 = tf.concat([self.continue_x, tf.reshape(self.embeddings, 
                                                        shape=[-1, self.field_size*self.embedding_size])],
                           axis=-1) 
        # deep part
        self.deep()
        # cross part
        self.cross()

        # concat
        concat_out = tf.concat([self.deep_out, self.cross_output], axis=-1)

        self.out = tf.layers.dense(concat_out, 
                                  self.num_classes,
                                  activation=tf.nn.sigmoid)
        self.logits = tf.sigmoid(self.out)

        # loss
        self.y_prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=-1)
        y = tf.one_hot(self.y, self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss, axis=0)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)

    def deep(self):
        self.deep_out = self.x0
        for i in range(len(self.deep_layer_sizes)):
            self.deep_out = tf.layers.dense(self.deep_out, 
                                         deep_layer_sizes[i],
                                         activation=tf.nn.relu)
            self.deep_out = tf.nn.dropout(self.deep_out, self.deep_keep_prob[i])


    def cross(self):
        self._x0 = tf.reshape(self.x0, shape=[-1, self.total_size, 1])
        x_l = self._x0
        for l in range(self.cross_layer_num):
            _xl = tf.matmul(self._x0, x_l, transpose_b=True)
            _x_l = tf.tensordot(_xl, self.__dict__.get("cross_w%s" %l), 1) + self.__dict__.get("cross_b%s" %l)
            # 将上一层的结果输入
            x_l = _x_l + x_l
            x_l = tf.nn.dropout(x_l, self.cross_keep_probs[l])

        self.cross_output = tf.reshape(x_l, [-1, self.total_size])

