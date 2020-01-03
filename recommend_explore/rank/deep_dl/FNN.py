#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：FNN.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import numpy as np
import tensorflow as tf

# num_features = [0] * 26
# with open('../data/featindex.txt') as fin:
#     for line in fin:
#         line = line.strip().split(':')
#         if len(line)>1:
#             f = int[line[0]]-1
#             num_features[f] += 1
# fnn_params = {
#         'field_sizes': num_features,
#         'embedding_size': 10,
#         'layer_sizes': [500, 1],
#         'learning_rate': 0.1,
#         'embed_l2': 0,
#         'layer_l2': [0, 0],
#     }

class FNN(object):
    def __init__(self, **kwargs):
        # feature length, 每个特征属于一个field
        self.field_sizes = kwargs.get("field_sizes")
        self.embedding_size = kwargs.get("embeding_size", 10)
        self.layer_sizes = kwargs.get("layer_sizes", [256, 128, 10])
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.layer_l2 = kwargs.get("layer_l2")
        self.embed_l2 = kwargs.get("embed_l2")

    def __call__(self):
        # init placeholder
        # 每个field 单独处理
        self.x = [tf.sparse_placeholder(dtype=tf.float32, shape=[None,None], name="x_%s" %i) for i in range(self.filed_size)]
        self.y = tf.placeholder(tf.float32, name="y")
        self.keep_drop = tf.placeholder(tf.float32, name="keep_prob")

        num_inputs = len(self.field_sizes)
        # init field embedding
        for i in range(num_inputs):
            # 这里的输入应该使用FM 进行embedding的结果, 这里选择随机初始化;
            embed_v = tf.get_variable("embed_%s" %i, shape=[self.field_sizes[i], self.embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            setattr(self, "embed_%s" %i, embed_v)

        # init nn weigth & bias
        node_input = num_inputs * embedding_size
        for i in range(len(layer_sizes)):
            w = tf.get_variable("w_%s" %i, shape=[node_input, layer_sizes[i]], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.Variable("b_%s" %i, shape=[layer_sizes[i]], 
                            initializer=tf.zeros_initializer(), dtype=tf.float32)
            setattr(self, "w_%s" %i, w)
            setattr(self, "b_%s" %i, b)
            node_input = layer_sizes[i]

        def build_graph():
            field_outs = list()
            for i in range(num_inputs):
                field_out = tf.sparse_tensor_dense_matmul(self.x[i], self.__dict__.get("embed_%s" %i))
                field_outs.append(field_out)
            f_out = tf.concat([field_outs], axis=1)
            xw = f_out

            for i in range(len(self.layer_sizes)):
                w_i = self.__dict__.get("w_%s" %i)
                b_i = self.__dict__.get("b_%s" %i)
                l_out = tf.matmul(f_out, w_i) + b_i
                l_out = tf.nn.relu(l_out)
                l_out = tf.nn.dropout(l_out, self.keep_drop)
                f_out = l_out

            f_out = tf.squeeze(f_out)
            self.y_prob = tf.sigmoid(f_out)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=f_out, labels=self.y))
            if self.layer_l2 is not None:
                self.loss += self.embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(self.layer_sizes)):
                    wi = self.__dict__.get('w_%d' % i)
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

