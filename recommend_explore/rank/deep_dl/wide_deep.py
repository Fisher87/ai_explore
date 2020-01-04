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
        self.category_field_size = kwargs.get("category_field_size")
        self.dense_feature_size = kwargs.get("dense_feature_size")
        self.cross_feature_size = kwargs.get("cross_feature_size")
        self.embedding_size = kwargs.get("embedding_size")
        self.layers_sizes = kwargs.get("layers_sizes")
        self.keep_drop = kwargs.get("keep_drop", 0.5)
        self.num_classes = kwargs.get("num_classes", 2)

    def __call__(self):
        # init placeholder
        self.category_x = tf.placeholder(tf.float32, shape=[None, self.category_feature_size], name="category_x")
        self.category_inx = tf.placeholder(tf.float32, shape=[None, self.category_field_size], name="category_inx")
        self.continue_x = tf.placeholder(tf.float32, shape=[None, self.continue_feature_size], name="continue_x")
        self.continue_inx = tf.placeholder(tf.float32, shape=[None, self.continue_field_size], name="continue_inx")
        self.cross_x = tf.placeholder(tf.float32, shape=[None, self.cross_feature_size], name="cross_x")
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="y")

        self.embedding_table = tf.get_variable("embed_%s" %i, 
                                            shape=[self.category_feature_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        ## wide
        def wide():
            # wide 部分使用的特征: cross product transformation生成的组合特征;
            wide_input = tf.concat([self.category_x, self.cross_x], axis=-1)
            # wide_out = tf.layers.dense(wide_input, self.y.shape[1], 
            #                           activation="sigmoid",
            #                           use_bias=True,
            #                           )
            wide_out = wide_input

            return wide_out

        def deep():
            # deep 部分使用的特征: 连续特征, embedding后的离散特征，item特征;
            embeddings = tf.nn.embedding_lookup(self.embedding_table, self.category_x)
            deep_input = tf.concat([tf.reshape(embeddings, [-1, self.category_feature_size*self.embedding_size]), 
                                   self.continue_x, self.category_inx, self.continue_inx], axis=-1)
            d = tf.layers.dense(deep_input, 50,
                               activation="relu", 
                               use_bias=True,
                               )
            d = tf.dropout(d, self.keep_drop)

            d = tf.layers.dense(d, 20,
                               activation="relu",
                               use_bias=True)
            deep_out = tf.dropout(d, self.keep_drop)

            return deep_out

        wide_out = wide()
        deep_out = deep()

        wd_in = tf.concat([wide_out, deep_out], axis=-1)
        self.logits = tf.layers.dense(wd_in, self.num_classes, 
                                    activation="sigmoid",
                                    name="wide_deep")
        self.y_prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=-1)
        y = tf.one_hot(self.y, self.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss, axis=0)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)


