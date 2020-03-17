#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ConvNet.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月04日
#   描    述：
#
#================================================================

import tensorflow as tf

class ConvNet(object):
    def __init__(self, 
                seq_length,
                vocab_size,
                embedding_dim,
                filter_sizes,
                filter_num,
                keep_prob, 
                class_size=2,
                l2_reg_lambda=0.001,
                learning_rate=0.001,
                hidden_size = 256,
                random_embedding=True):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.filter_num   = filter_num
        self.keep_prob = keep_prob
        self.class_size = class_size
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.hidden_size = hidden_size
        self.random_embedding = random_embedding

    @staticmethod
    def encoding(x, 
                 filter_sizes, 
                 embedding_dim,
                 filter_num,
                 k,
                 scope_name_prefix="conv_maxpooling-p"):
        """
        @param: x, 
        @param: filter_sizes, `list` type, e.g.[2,3,4].
        @param: embedding_dim, embedding size.
        @param: filter_num, each filter conv output size.
        @param: k, top_k maxpooling.

        return : a list of top_k_filter_maxpooling([batchsize, 2*filter_num]), 
                 list length is defined by filter_sizes length. 
        """
        conv_output = []
        for filter_size in filter_sizes:
            filter_shape = [filter_size, self.embedding_dim, 1, self.filter_num]
            with tf.name_scope("%s-%s" % (scope_name_prefix, filter_size)):
                filter_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_w")
                filter_b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]), name="filter_b")

                # tf.nn.conv2d(input,:     [batch, in_height, in_width, in_channels]
                #              filters,:   [filter_height, filter_width, in_channels, out_channels]
                #              strides,
                #              padding
                #              )
                conv = tf.nn.conv2d(x, 
                                    filter_w, 
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                conv_ = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # k-max pooling instead of a single max value.
                # k-max pooling means select more important chars/words, conv_[batch_size, seq_length, 1, filter_num],
                # so need to transpose `seq_length` to axis -1 ==> (transpose(conv_, [0,3,2,1]));
                #   tf.nn.top_k() return axis=-1 top k `values`` and `indices`
                # top_k_maxpooling -> [batchsize, k*filter_num]
                top_k_maxpooling = tf.reshape(tf.nn.top_k(tf.transpose(conv_, [0,3,2,1]), k=self.k), 
                                                  [-1, self.k*self.filter_num])
                conv_output.append(p_top_k_maxpooling)

        return conv_output

    @staticmethod
    def dropout(x, keep_prob=0.5):
        return tf.nn.dropout(x, keep_prob=keep_prob)

    def __call__(self):
        #init placeholder
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="premise")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name="hypothesis")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="drop_rate")

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            if self.random_embedding:
                embedding_table = self.get_variable(dtype=tf.float32, shape=[self.vocab_size, embedding_dim], 
                                                    name="embedding_table")
            else:
                # TODO
                embedding_table = load_embedding()

            p_embedding = tf.nn.embedding_lookup(embedding_table, self.p)
            h_embedding = tf.nn.embedding_lookup(embedding_table, self.h)

            p_embedding_expanded = tf.expand_dims(p_embedding, axis=-1)
            h_embedding_expanded = tf.expand_dims(h_embedding, axis=-1)

        # convolution + maxpool layer
        # feature extrace layer
        p_conv_output = self.encoding(p_embedding_expanded, self.filter_sizes, scope_name_prefix="conv-maxpooling-p")
        h_conv_output = self.encoding(h_embedding_expanded, self.filter_sizes, scope_name_prefix="conv-maxpooling-h")
        # concat all pooling info.
        # p_conv = tf.reshape(tf.concat(p_conv_ouput, axis=1), [-1, self.k*self.filter_num*len(self.filter_sizes)],
        #                    name="p_concat_conv")     # [batchsize, self.k*self.filter_num*self.filter_sizes]
        # h_conv = tf.reshape(tf.concat(h_conv_ouput, axis=1), [-1, self.k*self.filter_num*len(self.filter_sizes)],
        #                    name="h_concat_conv")

        # == is equal to above.
        p_conv = tf.concat(p_conv_ouput, axis=1, name="p_concat_conv")  # [batchsize, self.k*self.filter_num*self.filter_sizes]
        h_conv = tf.concat(h_conv_ouput, axis=1, name="h_concat_conv")  # [batchsize, self.k*self.filter_num*self.filter_sizes]

        # matching p and h; X_sim, compute cosine similarity.
        #   sim(x_p, x_h) = x_q.T * M * x_h
        with tf.name_scope("similarity"):
            sim_M = tf.get_variable("sim_M", shape=[p.shape[-1], p.shape[-1]], 
                                    initializer=tf.contrib.layers.xavier_initializer())
            _sim = tf.matmul(p_conv, sim_M) # [batchsize, self.k*self.filter_num*self.filter_sizes]
            _sim = tf.multiply(sim, h_conv)
            self.sim = tf.reduce_mean(sim, axis=1, keep_dims=True)  # [batchsize, 1]

        # concat all info.
        with name_scope("info_join"):
            # x_join = [x_q.T; x_sim; x_h.T; x_feat.T] (no X_feat).
            x_join = tf.concat([p_conv, self.sim, h_conv], axis=1, name="x_join")

        # fc layer
        # with name_scope("fc_layer"):
        #     x = tf.layers.dense(x_join, self.hidden_size, activation="relu")
        #     x = self.dropout(x, self.keep_prob)
        #     self.logits = tf.layers.dense(x, 2, activation="relu")

        # using l2 norm
        l2_loss = tf.constant(0.0)
        with name_scope("fc_layer"):
            fc_w = tf.get_variable("fc_w", 
                                  shape=[x_join.shape[-1], self.hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="fc_b")
            l2_loss += tf.nn.l2_loss(fc_w)
            hidden_out = tf.nn.relu(tf.nn.xw_plus_b(x_join, fc_w, fc_b, name="hidden_out"))
            hidden_out = self.dropout(hidden_out, self.keep_prob)

            logit_w = tf.get_variable("logit_w", 
                                  shape=[256, 2],
                                  initializer=tf.contrib.layers.xavier_initializer())
            logit_b = tf.Variable(tf.constant(0.1, shape=[2]), name="logit_b")
            l2_loss += tf.nn.l2_loss(logit_w)
            self.logits = tf.nn.relu(tf.nn.xw_plus_b(hidden_out, logit_w, logit_b, name="logit"))

        # loss 
        with tf.name_scope("loss"):
            y = tf.one_hot(self.y, self.class_size)
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
            self.loss = tf.reduce_mean(self.loss) + self.l2_reg_lambda*l2_loss

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.prediction = tf.argmax(self.logits, axis=1)
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        return self
