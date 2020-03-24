#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：DSSM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#       refer:https://github.com/terrifyzhao/text_matching/blob/master/dssm/graph.py
#
#================================================================

import pdb
import tensorflow as tf

class DSSM(object):
    def __init__(self, sequence_length, 
                       vocab_size, 
                       class_size = 2,
                       embedding_dim=128,
                       random_embedding=True,
                       hidden_units = 128,
                       learning_rate=0.001):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.embedding_dim = embedding_dim
        self.random_embedding = random_embedding
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

    def __call__(self):
        self.query = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name="query")
        self.doc = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name="doc")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")

        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if self.random_embedding:
                self.embedding_table = tf.get_variable(name="embedding", dtype=tf.float32, 
                                                       shape=[self.vocab_size, self.embedding_dim])
                # self.embedding_table = tf.Variable(
                #     tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                #     name="embedding")
            else:
                # TODO
                self.embedding_table = load_embedding()

            self.q_embedding = tf.nn.embedding_lookup(self.embedding_table, self.query)
            self.d_embedding = tf.nn.embedding_lookup(self.embedding_table, self.doc)

        # feature extract layer. DSSM using 3 fc layers.
        with tf.name_scope("fc_layer"):
            q_context_rep = self.fc(self.q_embedding)
            d_context_rep = self.fc(self.d_embedding)

            # refer https://github.com/terrifyzhao/text_matching/blob/master/dssm/graph.py
            pos_result = self.cosine(q_context_rep, d_context_rep)
            neg_result = 1 - pos_result
            self.logits = tf.concat([pos_result, neg_result], axis=1)
            
            # numclasses is 2.
        with tf.name_scope("output"):
            y = tf.one_hot(self.y, self.class_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
            self.optimier = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimier.minimize(self.loss, global_step=self.global_step)
            self.predictions = tf.argmax(self.logits, axis=1)
            self.correct_pred = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        return self
            
    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def fc(self, x):
        """
        fully connect layer
        """
        hidden_units = [256, 512, 128]
        for h_u in hidden_units:
            x = tf.layers.dense(x, h_u, activation="tanh")
            x = self.dropout(x)

        # prepare for compute cosine similarity.
        x = tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2]])

        return x

    def cosine(self, x1, x2):
        """
        cosine = y_qT * y_d / ||y_q||*||y_d||

        refer: 
            https://stackoverflow.com/questions/43357732/how-to-calculate-the-cosine-similarity-between-two-tensors
        """
        x1_norm = tf.norm(x1, axis=1, keepdims=True)
        x2_norm = tf.norm(x2, axis=1, keepdims=True)
        cosine = tf.reduce_sum(tf.multiply(x1, x2), axis=1, keepdims=True) \
                                    / (x1_norm * x2_norm)

        return cosine


