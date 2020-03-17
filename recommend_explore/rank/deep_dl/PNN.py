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
        '''
        feature_size: 特征值的个数;
        field_size:特征种类的个数;
        embedding_size:embeddeing 维度, 一般为FM算法预训练出的隐含向量表示的向量维度K;
        deep_layers: nn layers中每层的hidden size, 是一个list, e.g.:[128, 64, 10]
        learning_rate: 学习率;
        layer_l2: nn layers中每层对应的正则项权重值 lambda;
        embed_l2: embedding layer 对应的正则权重;
        use_inner: 是否使用inner product; 这里没有用到，只实现了inner product方式;
        '''
        self.feature_size = kwargs.get("feature_size")
        self.field_sizes = kwargs.get("field_sizes")
        self.embedding_size = kwargs.get("embedding_size")
        self.deep_layers = kwargs.get("deep_layers")
        self.learning_rate = kwargs.get('learning_rate')
        self.use_inner = kwargs.get("use_inner")
        self.layer_l2 = kwargs.get("layer_l2")  
        self.embed_l2 = kwargs.get("embed_l2")


    def init_weight(self, num_inputs, num_pairs):
        # init weights
        ## for embeddings
        w_feature_embeddings = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.1), 
                                           name="feature_embeddings")
        w_feature_bias = tf.get_variable("feature_bias", 
                                        initializer=tf.zeros_initializer(), dtype=tf.float32)
        setattr(self, "w_feature_embeddings", w_feature_embeddings)
        setattr(self, "w_feature_bias", w_feature_bias)

        ## for nn layer
        node_in = num_inputs * self.embedding_size + num_pairs
        for i in range(len(self.deep_layers)):
            wi = tf.get_variable('w_%s' %i, [node_in, self.deep_layers[i]], 
                                initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            bi = tf.Variable("b_%s" %i, shape=[self.deep_layers[i]], 
                            initializer=tf.zeros_initializer(), dtype=tf.float32)
            setattr(self, "w_%s" %s, wi)
            setattr(self, "b_%s" %s, bi)
            node_in = self.deep_layers[i]

    def __call__(self):
        self.feature_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="feature_index")
        self.feature_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name="feature_value")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        num_inputs = len(self.field_sizes)
        num_pairs = int(0.5 * (num_inputs * (num_inputs-1)))
        
        def build_graph():
            self.init_weight(num_inputs, num_pairs)
            self.embeddings = tf.nn.embedding_lookup(self.__dict__.get("w_feature_embeddings", self.feature_index), name="embed")
            feature_value = tf.reshape(self.feature_value, shape=[-1, len(self.field_sizes), 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)
            
            # linear part
            self.lz = tf.reshape(self.embeddings, [-1, num_inputs, self.embedding_size])
            lz = tf.reshape(self.lz, [-1, num_inputs*self.embedding_size])

            # inner product
            row = []
            col = []
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    row.append(i)
                    col.append(j)

            # batch * pair * k
            p = tf.transpose(
                # pair * batch * k
                tf.gather(
                    # num_inputs * batch * k
                    tf.transpose(self.lz, [1, 0, 2]), 
                    row
                ),
                [1, 0, 2]
            )
            
            # batch * pair * k
            q = tf.transpose(
                tf.gather(
                    tf.transpose(self.lz, [1, 0, 2]),
                    col
                ),
                [1, 0, 2]
            )

            p = tf.reshape(p, [-1, num_pairs, embed_size])
            q = tf.reshape(q, [-1, num_pairs, embed_size])
            self.lp = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
            
            out = tf.concat([lz, self.lp], 1)  #(batch, num_inputs * self.embedding_size + num_pairs)

            # nn layer
            for i in range(len(layer_sizes)):
                wi = self.__dict__.get("w_%s" %i)
                bi = self.__dict__.get("b_%s" %i)
                out = tf.nn.dropout(
                    tf.nn.relu(tf.matmul(out, wi) + bi), 
                    self.keep_prob
                )

            out = tf.squeeze(out)
            self.y_prob = tf.sigmoid(out)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.y)
            )

            # 模型参数正则
            if self.layer_l2 is not None:
                self.loss += self.embed_l2 * tf.nn.l2_loss(lz)
                for i in range(len(self.deep_layers)):
                    wi = self.__dict__.get("w_%s" %i)
                    self.loss += self.layer_l2[i] * tf.nn.l2_loss(wi)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losss)
