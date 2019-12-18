#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：LR.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import tensorflow as tf

class LR(object):
    def __init__(self, learning_rate, feature_num, class_size):
        self.learning_rate = learning_rate
        self.feature_num = feature_num
        self.class_size = class_size

    def __call__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_num], name="x")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")

        self.w = tf.Variable(tf.zeros([self.feature_num, self.class_size]))
        self.b = tf.Variable(tf.zeros([self.class_size]))

        self.logits = tf.matmul(self.x, self.w) + self.b

        y = tf.one_hot(self.y, self.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.predict = tf.argmax(self.logits, axis=-1)
        correct_precition = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_precition, tf.float32), name="acc")

