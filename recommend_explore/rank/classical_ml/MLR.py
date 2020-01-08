#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：MLR.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月08日
#   描    述：
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.layers.l1_regularizer as l1

from sklearn.metrics import roc_auc_score

class MLR(object):
    def __init__(self, **kwargs):
        # 分组数
        self.m = kwargs.get('m')
        self.feature_size = kwargs.get("featuer_size")
        self.learning_rate = kwargs.get("learning_rate")
        self.keep_prob = kwargs.get("keep_prob")

    def __call__(self):
        # init placeholder
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size], name="x")
        self.y = tf.placehoder(dtype=tf.float32, shape=[None], name="y")

        # divide layer
        with tf.variable_scope("divide_layer"):
            divide = tf.layers.dense(self.x, 
                                     self.m,
                                     activation=tf.nn.softmax,
                                     kernel_regularizer=l1(0.01)
                                     )
            divide = tf.nn.dropout(divide, self.keep_prob)

        # fitting layer
        with tf.variable_scope("fitting_layer"):
            fit = tf.layers.dense(self.x,
                                  self.m,
                                  activation=tf.nn.sigmoid,
                                  kernel_regularizer=l1(0.01)
                                 )
            fit = tf.nn.dropout(fit, self.keep_prob)

        pred = tf.reshape(tf.reduce_sum(divide*fit, -1), [-1, 1])

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.y)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss)
        self.auc = roc_auc_score(self.y, pred)


