#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：wide_deep.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#         reference:
#                [1]. https://github.com/tensorflow/models/tree/master/official/r1/wide_deep
#
#================================================================

class WideDeep(object):
    def __init__(self, **kwargs):
        self.category_feature_size = kwargs.get("category_feature_size")
        self.dense_feature_size = kwargs.get("dense_feature_size")
        self.embedding_size = kwargs.get("embedding_size")
        self.layers_sizes = kwargs.get("layers_sizes")

    def __call__(self):
        # init placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.category_feature_size], name="x")
        self.dense_x = tf.placeholder(tf.float32, shape=[None, self.dense_feature_size], name="dense_x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")

        self.embedding_table = tf.get_variable("embed_%s" %i, 
                                            shape=[self.category_feature_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # set weights

        ## wide
        def wide():
            embedding = tf.nn.embedding_lookup(self.embedding_table, self.x)
            x_input = tf.reshape(embedding, [-1, self.category_feature_size*self.embedding_size])
            x_input = tf.concat([x_input, self.dense_x], axis=-1)
            for i in range(len(layers_sizes)):
                pass

        def deep():
            pass

