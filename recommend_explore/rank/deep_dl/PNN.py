#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：PNN.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#          reference: [1]. https://cloud.tencent.com/developer/article/1164785
#                     [2]. https://github.com/Atomu2014/product-nets/blob/master/python/models.py
#
#================================================================

import tensorflow as tf

class PNN(object):
    def __init__(self, **kwargs):
        self.feature_size = kwargs.get("feature_size")
        self.field_sizes = kwargs.get("field_sizes")
        self.embedding_size = kwargs.get("embedding_size")
        self.deep_layers = kwargs.get("deep_layers")
        self.learning_rate = kwargs.get('learning_rate')
        self.use_inner = kwargs.get("use_inner")


    def __call__(self):
        self.feature_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="feature_index")
        self.feature_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name="feature_value")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # init weights
        ## for embeddings
        w_feature_embeddings = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.1), 
                                           name="feature_embeddings")
        w_feature_bias = tf.get_variable("feature_bias", 
                                        initializer=tf.zeros_initializer(), dtype=tf.float32)
        setattr(self, "w_feature_embeddings", w_feature_embeddings)
        setattr(self, "w_feature_bias", w_feature_bias)

        ## for product layer
        ### product linear part
        p_l_w = tf.Variable(tf.random_normal[])

        num_inputs = len(self.field_sizes)
        if self.use_inner:
            pq_w_inner = tf.Variable(tf.random_normal([num_inputs, ]))
        else:
            pass

        
        def build_graph():
            self.embeddings = tf.nn.embedding_lookup(self.__dict__.get("w_feature_embeddings", self.feature_index), name="embed")
            feature_value = tf.reshape(self.feature_value, shape=[-1, len(self.field_sizes), 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)
            
            # linear part
            linear_out = []
            for i in range(len(num_inputs)):
                l_o = tf.reduce_sum(tf.multiply(self.embeddings, self.__dict__.get("w_product_linear")))
                linear_out.append()






