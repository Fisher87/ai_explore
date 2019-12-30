#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：FM.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import tensorflow as tf

class FM(object):
    def __init__(self, 
                 feature_size, 
                 class_num,
                 hidden_size,
                 l1_regularization_strength,
                 l2_regularization_strength,
                 learning_rate=0.01):
        self.feature_size = feature_size
        self.class_num = class_num
        # number of laten factors 
        self.hidden_size   = hidden_size
        self.learning_rate = learning_rate
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength

    def __call__(self):
        # init placeholder
        self.X = tf.sparse_placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        # build graph
        def build_graph():
            with tf.variable_scope("linear_layer"):
                w = tf.get_variable("w", shape=[self.feature_size, self.class_num], 
                                   initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
                b = tf.get_variable("b", shape=[self.class_num], 
                                   initializer=tf.zeros_initializer())
                self.linear_output = tf.matmul(self.X, w) + b

            with tf.variable_scope("interaction_layer"):
                v = tf.get_variable("v", shape=[self.feature_size, self.hidden_size], 
                                   initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
                # ∑_{i=1}^n∑_{j=i+1}^n<V_i,V_j>x_ix_j = 
                #                                     = 1/2 ∑{f=1}^k(( ∑{i=1}^n*v_if*xi)2−∑{i=1}^n*v_{if}^2*x_i^2))
                # self.interaction_terms = tf.multiply(0.5,
                #                                  tf.reduce_sum(
                #                                      tf.subtract(
                #                                          tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
                #                                          tf.sparse_tensor_dense_matmul(tf.pow(self.X, 2), tf.pow(v, 2))),
                #                                      1, keep_dims=True))
                pow_part = tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2)
                square_mul_part = tf.sparse_tensor_dense_matmul(tf.pow(self.X, 2), tf.pow(v, 2)) 
                subtract_v = tf.substract(pow_part, square_mul_part)
                self.interaction_terms = tf.multiply(0.5, tf.reduce_sum(subtract_v, axis=1, keep_dims=True))

        build_graph()
        y_out = tf.add(self.linear_output + self.interaction_terms)
        self.logits = tf.nn.softmax(y_out)

        y = tf.one_hot(self.y, self.class_num)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', self.loss)

        # add train opt
        self.train_opt = tf.train.FtrlOptimizer(self.learning_rate, 
                                              l1_regularization_strength=self.reg_l1,
                                              l2_regularization_strength=self.reg_l2).minimize(self.loss)

        # accuracy
        predict_correct = tf.equal(tf.cast(tf.argmax(self.logits), tf.float32), self.y)
        self.accuracy = tf.readuce_mean(tf.cast(predict_correct, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)




