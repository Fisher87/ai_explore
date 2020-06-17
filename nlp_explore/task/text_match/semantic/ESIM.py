#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：ESIM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#
#================================================================

import tensorflow as tf

class ESIM(object):
    def __init__(self, 
                 seq_length, 
                 num_classes,
                 embedding_dim,
                 vocab_size, 
                 hidden_size,
                 learning_rate=0.001,
                 random_embedding=True):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size 
        self.learning_rate = learning_rate
        self.random_embedding = random_embedding
        self.build_graph()

    def encoding(self, x, seq_len=None, reuse=False, scope_name="encoding"):
        with tf.variable_scope(scope_name, reuse=reuse):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

            return tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, 
                                                   bw_lstm_cell, 
                                                   x, 
                                                   sequence_length=seq_len,
                                                   dtype=tf.float32)

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
        
    def build_graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name="input_x")
        self.input_x_mask = tf.cast(tf.math.equal(self.input_x, 0), tf.float32)
        self.input_y = tf.placeholder(tf.int32, [None, self.seq_length], name="input_y")
        self.input_y_mask = tf.cast(tf.math.equal(self.input_y, 0), tf.float32)
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name="label")

        # real length
        # self.input_x_len = tf.placeholder(tf.int32, [None], name="p_len")
        # self.input_y_len = tf.placeholder(tf.int32, [None], name="h_len")
        self.input_x_len = tf.count_nonzero(self.input_x, axis=1, name="x_len")
        self.input_y_len = tf.count_nonzero(self.input_y, axis=1, name="y_len")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            if self.random_embedding:
                self.embedding_table = tf.get_variable(name="embedding", dtype=tf.float32, 
                                                       shape=[self.vocab_size, self.embedding_dim])
            else:
                # TODO
                self.embedding_table = load_embedding()

            self.embedding_p = tf.nn.embedding_lookup(self.embedding_table, self.input_x)
            self.embedding_h = tf.nn.embedding_lookup(self.embedding_table, self.input_y)

        # input encoding, `BiLSTM`
        (p_f, p_b), _ = self.encoding(self.embedding_p, self.input_x_len, reuse=tf.AUTO_REUSE, scope_name="lstm_p")
        (h_f, h_b), _ = self.encoding(self.embedding_h, self.input_y_len, reuse=tf.AUTO_REUSE, scope_name="lstm_h")

        p = self.dropout(tf.concat([p_f, p_b], axis=2))  # [batch_size, max_time, hidden_size*2]
        h = self.dropout(tf.concat([h_f, h_b], axis=2))  # [batch_size, max_time, hidden_size*2]

        # local inference modeling layer
        with tf.name_scope("local_inference_modeling"):
            # e_ij = a_i.T * b_j
            e = tf.matmul(p, tf.transpose(h, [0, 2, 1])) # [batch_size, max_time(p), max_time(h)]

            # NOTE:compute attention need using mask;
            # refer:https://github.com/terrifyzhao/text_matching/commit/002758625b16366fc4985c8dd6fddfb4ccfcf9ff
            # tf.nn.softmax :default axis=-1
            #      a_attention_i = \sum_{j=1}^{len(h_j)}( exp(e_ij) / \sum_{k=1}^{len(h_j)(exp(e_ik)) * h_j} )
            #      b_attention_j = \sum_{i=1}^{len(p_i)}( exp(e_ij) / \sum_{k=1}^{len(p_i)(exp(e_kj)) * p_i} )
            a_attention = tf.nn.softmax(e + tf.tile(tf.expand_dims(self.input_y_mask*(-2**32 + 1),1), [1, tf.shape(e)[1],1]))
            b_attention = tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1]) + 
                                             tf.tile(tf.expand_dims(self.input_x_mask*(-2**32 + 1),1), 
                                                     [1, tf.shape(tf.transpose(e, perm=[0, 2, 1]))[1],1])
                                       )
            a = tf.matmul(a_attention, h)
            b = tf.matmul(b_attention, p)

            # 对encoding值与加权encoding值进行差异值计算(对位相减与对位相乘)，并进行拼接
            m_a = tf.concat([a, p, a-p, tf.multiply(a, p)], axis=2)
            m_b = tf.concat([b, h, b-h, tf.multiply(b, h)], axis=2)

        # inference composition, use same LSTM layer
        (a_f, a_b), _ = self.encoding(m_a, self.input_x_len, reuse=tf.AUTO_REUSE, scope_name="lstm_infer")
        (b_f, b_b), _ = self.encoding(m_b, self.input_y_len, reuse=tf.AUTO_REUSE, scope_name="lstm_infer")

        a = tf.concat([a_f, a_b], axis=2)
        b = tf.concat([b_f, b_b], axis=2)

        a = self.dropout(a)
        b = self.dropout(b)

        a_avg = tf.reduce_mean(a, axis=2)
        b_avg = tf.reduce_mean(b, axis=2)

        a_max = tf.reduce_max(a, axis=2)
        b_max = tf.reduce_max(b, axis=2)

        v = tf.concat([a_avg, a_max, b_avg, b_max], axis=1)

        # prediction layer
        v = tf.layers.dense(v, 512, activation="tanh")
        v = self.dropout(v)
        # y = tf.one_hot(self.label, self.num_classes)
        self.logits = tf.layers.dense(v, self.num_classes, activation="tanh")
        self.prob = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        self.loss = tf.reduce_mean(loss, axis=0)
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), 
                                          tf.cast(tf.argmax(self.label, axis=1), tf.int32))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)

