#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：DRCN.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月21日
#   描    述：
#
#================================================================

import tensorflow as tf

class DRCN(object):
    def __init__(self, **kwargs):
        self.seq_c_len = kwargs.get("seq_c_len")
        self.seq_w_len = kwargs.get("seq_w_len")
        self.class_size= kwargs.get("class_size")
        self.char_vocab_len = kwargs.get("char_vocab_len")
        self.char_embedding_dim = kwargs.get("char_embedding_dim")
        self.word_vocab_len = kwargs.get("word_vocab_len")
        self.word_embedding_dim = kwargs.get("word_embedding_dim")
        self.word_embedding_initializer = kwargs.get("word_embedding_initializer")
        self.learning_rate = kwargs.get("learning_rate")

    def __call__(self):
        self.p_c = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_c_len, self.char_embedding_dim],
                                 name="p_c")
        self.h_c = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_c_len, self.char_embedding_dim],
                                 name="h_c")
        self.p_w = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_w_len, self.word_embedding_dim],
                                 name="p_w")
        self.h_w = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_w_len, self.word_embedding_dim],
                                 name="h_w")
        self.p_w_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_w_len, self.word_embedding_dim], 
                                     name="p_w_vec")
        self.h_w_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_w_len, self.word_embedding_dim], 
                                     name="h_w_vec")
        self.same_word = tf.placeholder(name="same_word", shape=[None], dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
        self.keep_prob_embed = tf.placeholder(name='keep_prob_embed', dtype=tf.float32)
        self.keep_prob_fully = tf.placeholder(name="keep_prob_fully", dtype=tf.float32)
        self.bn_training = tf.placeholder(name="bn_training", dtype=tf.bool)

        self.char_embedding_table = tf.get_variable(name="char_embedding_table",
                                                    shape=[self.char_vocab_len, self.char_embedding_dim]
                                                    dtype=tf.float32
                                                   )
        self.word_embedding_table = tf.get_variable(name="word_embedding_table",
                                                    initializer=self.word_embedding_initializer)

        # word representation layer
        # char embedding
        p_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.p_c)
        h_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.h_c)
        # word embedding
        p_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.p_w)
        h_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.h_w)

        same_word = tf.expand_dims(
                        tf.expand_dims(self.same_word, axis=-1), axis=-1) #[batch, 1, 1]
        same_word = tf.tile(same_word, [1, 15, 1])

        p = tf.concat([p_char_embedding, p_word_embedding, self.p_w_vec, same_word], axis=-1)
        h = tf.concat([h_char_embedding, h_word_embedding, self.h_w_vec, same_word], axis=-1)
        p = tf.nn.dropout(p, self.keep_prob_embed)
        h = tf.nn.dropout(h, self.keep_prob_embed)

        # attentively connected RNN
        for i in range(4):
            p_state, h_state = p, h
            for j in range(5):
                with tf.variable_scope(f"p_lstm_{i}_{j}", reuse=None):
                    p_state, _ = self.bilstm(tf.concat(p_state, axis=-1))
                with tf.variable_scope(f"h_lstm_{i}_{j}", reuse=None):
                    h_state, _ = self.bilstm(tf.concat(h_state, axis=-1))

                p_state = tf.concat(p_state, axis=-1)
                h_state = tf.concat(h_state, axis=-1)

                # attention
                cosine = tf.divide(
                    tf.matmul(p_state, tf.transpose(h_state, [0,2,1])),
                    tf.norm(p_state, axis=-1, keep_dims=True)*tf.norm(h_state, axis=-1, keep_dims=True)
                )
                att_matrix = tf.nn.softmax(cosine)
                p_attention = tf.matmul(att_matrix, h_state)
                h_attention = tf.matmul(att_matrix, p_state)

                # dense net
                p = tf.concat([p, p_state, p_attention], axis=-1)
                h = tf.concat([h, h_state, h_attention], axis=-1)

        # auto encoder
        p = tf.layers.dense(p, 200)
        h = tf.layers.dense(h, 200)

        # interaction and prediction layer
        add = p + h
        sub = p - h
        norm= tf.norm(sub, axis=-1)
        out = tf.concat([p, h, add, sub, tf.expand_dims(norm, axis=-1)], axis=-1)
        out = tf.reshape(out, shape=[-1, out.shape[1] * out.shape[2]])
        out = tf.nn.dropout(out, self.keep_prob_fully)

        out = tf.layers.dense(out, 1000, activation='relu')
        out = tf.layers.batch_normalization(out, training=self.bn_training)
        out = tf.layers.dense(out, 1000, activation='relu')
        out = tf.layers.batch_normalization(out, training=self.bn_training)
        out = tf.layers.dense(out, 1000)
        out = tf.layers.batch_normalization(out, training=self.bn_training)
        self.logits = tf.layers.dense(out, self.class_size)

        # training
        y = tf.one_hot(self.y, self.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)

        # batch_normalization计算均值和方差
        # 1.
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # 2.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([self.train_op, update_ops])

        self.predict = tf.argmax(self.logits, axis=1)
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def bilstm(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_size)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

