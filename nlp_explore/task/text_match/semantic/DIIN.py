#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：DIIN.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月25日
#   描    述：
#           reference : [1]. git@github.com:YichenGong/Densely-Interactive-Inference-Network.git
#                       [2]. git@github.com:terrifyzhao/text_matching.git
#                      
#
#================================================================

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from utils.general import _linear
from utils.general import flatten
from utils.general import reconstruct

class DIIN(object):
    def __init__(self, **kwargs):
        self.seq_len = kwargs.get('seq_len')
        self.max_char_len = kwargs.get("max_char_len")
        self.char_vocab_len = kwargs.get("char_vocab_size")
        self.char_embedding_dim = kwargs.get("char_embedding_dim")
        self.max_word_len = kwargs.get("max_word_len")
        self.embedding_dim = kwargs.get('embedding_dim')
        self.hidden_size = kwargs.get("hidden_size")
        self.learning_rate = kwargs.get("learning_rate")
        self.word_embedding = kwargs.get("word_embedding", None)
        self.dense_net_first_scale_down_ratio = kwargs.get("dense_down_ratio", 0.3)
        self.dense_growth_rate = kwargs.get("dense_g", 12)
        self.dense_net_transition_rate = kwargs.get("dense_transition_rate", 0.5)

    def self_attention(self, x, is_train=True, squeeze=False):
        with tf.variable_scope("attention"):
            l_x = x.shape[1] 
            x_aug_1 = tf.tile(tf.expand_dims(x, axis=2), [1, 1, l_x, 1])
            x_aug_2 = tf.tile(tf.expand_dims(x, axis=1), [1, l_x, 1, 1])
            new_x = x_aug_1 * x_aug_2
            flat_args = [x_aug_1, x_aug_2, new_x]
            flat_args = [flatten(arg, 1) for arg in flat_args]
            flat_args = [tf.cond(is_train, lambda: self.dropout(arg), lambda: arg) for 
                            arg in flat_args]
            flat_out = _linear(flat_args, 1, False)
            out = reconstruct(flat_out, x_aug_1, 1)
            if squeeze:
                out = tf.squeeze(out, [len(x_aug_1.get_shape().as_list()) - 1])

            return out

    def softsel(self, x, att, name="softsel"):
        with tf.name_scope(name):
            att = tf.nn.softmax(att)
            # x_rank = len(x.get_shape().as_list())
            # out = tf.reduce_sum(tf.expand_dims(a, -1)*x, x_rank-2)
            x_att = tf.matmul(att, x)
            return x_att

    def fuse_gate(self, x, x_att):
        x_concat = tf.concat((x, x_hat), axis=-1)
        z = tf.nn.tanh(tf.einsum("ijk,kl->ijl", x_concat, self.gate_w1) + self.gate_b1)
        r = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", x_concat, self.gate_w2) + self.gate_b2)
        f = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", x_concat, self.gate_w3) + self.gate_b3)
        res = tf.multiply(r, x) + tf.multiply(f, z)

        return res

    def dense_net(self, v):
        filters = int(v.shape[-1] * self.dense_net_first_scale_down_ratio)
        v_in = tf.layers.conv2d(v, filters=filters, kernel_size=(1, 1))
        for _ in range(3):
            for _ in range(8):
                v_out = tf.layers.conv2d(v_in,
                                         filters=self.dense_growth_rate,
                                         kernel_size=(3, 3),
                                         padding='SAME',
                                         activation='relu')
                v_in = tf.concat((v_in, v_out), axis=-1)
            transition = tf.layers.conv2d(v_in,
                                          filters=int(v_in.shape[-1].value * self.dense_net_transition_rate),
                                          kernel_size=(1, 1))
            transition_out = tf.layers.max_pooling2d(transition,
                                                     pool_size=(2, 2),
                                                     strides=2)
            v_in = transition_out
        return v_in


    def encode(self, x):
        attention = self.self_attention(x, is_train=True, squeeze=True)
        x_att = self.softsel(x, attention)
        out = self.fuse_gate(x, x_att)

        return x_att


    def __call__(self):
        # init placeholder
        self.p_c = placeholder(name='p', shape=(None, self.max_char_len), dtype=tf.int32)
        self.h_c = tf.placeholder(name='h', shape=(None, self.max_char_len), dtype=tf.int32)
        # self.p_w = tf.placeholder(name='p_word', shape=(None, self.max_word_len), dtype=tf.int32)
        # self.h_w = tf.placeholder(name='h_word', shape=(None, self.max_word_len), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.char_embedding_table = tf.get_variable(name='char_embed', shape=(self.char_vocab_len, self.char_embedding_dim), 
                                                    dtype=tf.float32)
        self.word_embedding_table = tf.get_variable(name='word_embed', initializer=word_embedding, dtype=tf.float32, 
                                                    trainable=False)
        self.self_w = tf.get_variable(name='self_w', shape=(self.hidden_size * 3, 30))
        self.gate_w1 = tf.get_variable(name='gate_w1', shape=(self.hidden_size * 2, self.hidden_size))
        self.gate_w2 = tf.get_variable(name='gate_w2', shape=(self.hidden_size * 2, self.hidden_size))
        self.gate_w3 = tf.get_variable(name='gate_w3', shape=(self.hidden_size * 2, self.hidden_size))
        self.gate_b1 = tf.get_variable(name='gate_b1', shape=(self.hidden_size,))
        self.gate_b2 = tf.get_variable(name='gate_b2', shape=(self.hidden_size,))
        self.gate_b3 = tf.get_variable(name='gate_b3', shape=(self.hidden_size,))


        # embedding layer
        ## char embedding
        ## 原文中使用的是CNN进行char embedding，这里直接使用查表方式;
        ## 字embedding
        p_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.p_c)
        h_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.h_c)

        ## word embedding 
        ## 使用预训练好的word embedding；
        ## 该示例中暂时不用词embedding
        # p_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.p_w)
        # h_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.h_w)

        ## 原文中还使用了POS、exact emedding, 此处暂时不加;

        # concat: 在axis=-1维度进行concat, 例如pos信息等, 
        # 保持embedding_dim维度相同;
        # 的结果保持是(b, seq_len, *) 的维度;
        # p = tf.concat([p_char_embedding, p_word_embedding], axis=-1)
        # h = tf.concat([h_char_embedding, h_word_embedding], axis=-1)
        p = p_char_embedding
        h = h_char_embedding

        # highway
        p = self.highway(p)
        h = self.highway(h)

        # encode layer
        ## self attention 的另外一种方式;
        p = self.dropout(p)
        h = self.dropout(h)
        with tf.variable_scope('p_encode', reuse=None):
            p = self.encode(p)
        with tf.variable_scope('h_encode', reuse=None):
            h = self.encode(h)

        # interaction layer
        i_matrix = tf.multiply(tf.expand_dims(p, axis=2), tf.expand_dims(h, axis=1))

        # feature extraction
        dense_net_out = self.dense_net(i_matrix)
        dense_out = self.dropout(dense_net_out)

        # output layer
        dense_out = tf.reshape(dense_out, shape=(-1, dense_out.shape[1] * dense_out.shape[2] * dense_out.shape[3]))
        out = tf.layers.dense(dense_out, 256)
        out = self.dropout(out)
        self.logits = tf.layers.dense(out, 2)

        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

