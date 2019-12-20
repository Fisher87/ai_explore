#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：BIMPM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月12日
#   描    述：
#
#================================================================

import tensorflow as tf

class Bimpm(object):
    def __init__(self, seq_len, word_embedding_dim, char_embedding_dim, char_vocab_len
                 char_hidden_size, num_perspective, learning_rate):
        self.seq_len = seq_len
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_vocab_len = char_vocab_len
        self.char_hidden_size = char_hidden_size
        self.num_perspective= num_perspective
        self.learning_rate = learning_rate

    def lstm(self, x, scope="lstm"):
        with tf.variable_scope(scope, reuse=None):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden_size)
            return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def bilstm(self, x, scope="bilstm", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden_size)

            return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def dropout(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def full_matching(self, metric, vec, w):
        w = tf.expand_dims(tf.expand_dims(w, 0), 2)
        metric = w * tf.stack([metric]*self.num_perspective, axis=1)
        vec = w * tf.stack([vec]*self.num_perspective, axis=1)

        m = tf.matmul(metric, tf.transpose(vec, [0,1,3,2]))
        n = tf.norm(metric, axis=3, keep_dims=True) * tf.norm(vec, axis=3, keep_dims=True)
        cosine = tf.transpose(tf.divide(m, n), perm=[0,2,3,1])

        return cosine

    def maxpool_matching(self, metric1, metric2, w):
        cosine = self.full_matching(metric1, metirc2)
        max_value = tf.reduce_max(cosine, axis=2, keep_dims=True)

        return max_value

    def attentive_match(self, metirc1, metirc2, w):
        cosine = tf.matmul(metric1, tf.transpose(metric2, [0,2,1]))
        norm = tf.norm(metric1, axis=-1, keep_dims=True) * tf.transpose(
                                   tf.norm(metric2, axis=-1, keep_dims=True), perm=[0,2,1])
        cosine = tf.divide(cosine, norm)

        metric1_att = tf.matmul(cosine, metric2)
        metric2_att = tf.matmul(cosine, metric1)

        metric1_mean = tf.divide(metric1_att, tf.reduce_sum(cosine, axis=2, keep_dims=True))
        metric2_mean = tf.divide(metric2_att, tf.reduce_sum(cosine, axis=2, keep_dims=True))

        metric1_att_mean = self.full_matching(metric1, metric1_mean, w)
        metric2_att_mean = self.full_matching(metric2, metric2_mean, w)

        return metric1_att, metric2_att, metric1_att_mean, metric2_att_mean

    def __call__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name="p")
        self.h = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name="h")
        self.p_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.word_embedding_dim], 
                                    name="p_word")
        self.h_vec = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.word_embedding_dim], 
                                    name="h_word")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")
        self.embed = tf.get_variable(dtype=tf.float32, shape=[self.char_vocab_len, self.char_embedding_dim],
                                    name="char_embed_table")
        for i in range(1,9):
            setattr(self, f'w{i}', 
                    tf.get_variable(dtype=tf.float32, shape=[self.num_perspective, self.char_hidden_size],
                                    name=f'w{i}')
                   )

        # input layer
        with tf.device("/cpu:0"), tf.name_scope("char_embedding"):
            p_char_embedding = tf.nn.embedding_lookup(self.embed, self.p)
            h_char_embedding = tf.nn.embedding_lookup(self.embed, self.h)

        p_output, _ = self.lstm(p_char_embedding, scope="lstm_p")
        h_output, _ = self.lstm(h_char_embedding, scope="lstm_h")
        ## concat char_embedding & word_embedding
        p_embedding = tf.concat([self.p_vec, p_output], axis=-1)
        h_embedding = tf.concat([self.h_vec, h_output], axis=-1)
        p_embedding = self.dropout(p_embedding)
        h_embedding = self.dropout(h_embedding)

        # context representation layer
        (p_fw, p_bw), _ = self.bilstm(p_embedding, scope="bilstm_p", reuse=tf.AUTO_REUSE)
        (h_fw, h_bw), _ = self.bilstm(h_embedding, scope="bilstm_h", reuse=tf.AUTO_REUSE)

        p_fw = self.dropout(p_fw)
        p_bw = self.dropout(p_bw)
        h_fw = self.dropout(h_fw)
        h_bw = self.dropout(h_bw)

        # match layer
        ## 1. full match
        p_full_fw = self.full_matching(p_fw, tf.expand_dims(h_fw[:,-1,:], axis=1), self.w1)
        h_full_fw = self.full_matching(h_fw, tf.expand_dims(p_fw[:,-1,:], axis=1), self.w1)
        p_full_bw = self.full_matching(p_bw, tf.expand_dims(h_bw[:,0,:], axis=1), self.w2)
        h_full_bw = self.full_matching(h_bw, tf.expand_dims(p_bw[:,0,:], axis=1), self.w2)

        ## 2. maxpooling match
        maxpool_fw = self.maxpool_matching(p_fw, h_fw, self.w3)    # [batch, l_p, 1, num_perspective]
        maxpool_bw = self.maxpool_matching(p_bw, h_bw, self.w4)    # [batch, l_p, 1, num_perspective]

        ## 3. attentvie match
        p_att_fw, h_att_fw, p_attmean_fw, h_attmean_fw = self.attentive_match(p_fw, h_fw, self.w5)
        p_att_bw, h_att_bw, p_attmean_bw, h_attmean_bw = self.attentive_match(p_bw, h_bw, self.w6)

        ## 4. max attentive match
        p_max_fw = tf.reduce_max(p_att_fw, axis=-1, keep_dims=True)
        p_max_bw = tf.reduce_max(p_att_bw, axis=-1, keep_dims=True)
        h_max_fw = tf.reduce_max(h_att_fw, axis=-1, keep_dims=True)
        h_max_bw = tf.reduce_max(h_att_bw, axis=-1, keep_dims=True)

        p_attmax_fw = self.full_matching(p_fw, p_max_fw, self.w7)
        h_attmax_fw = self.full_matching(h_fw, h_max_fw, self.w7)
        p_attmax_bw = self.full_matching(p_bw, p_max_bw, self.w8)
        h_attmax_bw = self.full_matching(h_bw, h_max_bw, self.w8)

        ## 5. concat
        mv_p = tf.concat(
            [p_full_fw, maxpool_fw, p_attmean_fw, p_attmax_fw,
             p_full_bw, maxpool_bw, p_attmean_bw, p_attmax_bw],
            axis=2
        )
        mv_h = tf.concat(
            [h_full_fw, maxpool_fw, h_attmean_fw, h_attmax_fw,
             h_full_bw, maxpool_bw, h_attmean_bw, h_attmax_bw],
            axis=2
        )
        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        mv_p = tf.reshape(mv_p, shape=[-1, mv_p.shape[1], mv_p.shape[2]*mv_p.shape[-1]])
        mv_h = tf.reshape(mv_h, shape=[-1, mv_h.shape[1], mv_h.shape[2]*mv_h.shape[-1]])

        # aggregation layer
        (p_f_last, p_b_last), _ = self.bilstm(mv_p, scope="bilstm_aggregation_p", reuse=tf.AUTO_REUSE)
        (h_f_last, h_b_last), _ = self.bilstm(mv_h, scope="bilstm_aggregation_h", reuse=tf.AUTO_REUSE)
        x = tf.concat([p_f_last, p_b_last, h_f_last, h_b_last], axis=-1)
        x = tf.reshape(x, shape=[-1, x.shape[1]*x.shape[-1]])
        x = self.dropout(x)

        # prediction layer
        x = tf.layers.dense(x, 1024, activation="tanh")
        x = self.dropout(x)
        x = tf.layers.dense(x, 512)
        self.logits = tf.layers.dense(x, 2)

        y = tf.one_hot(self.y, 2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.predict  = tf.argmax(self.logits, axis=-1, name="predict")
        correct_predict = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name="acc")

