#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：DIIN.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月25日
#   描    述：
#
#================================================================

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

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
            flat_out = rnn_cell_impl._linear(flat_args, 1)
            out = reconstruct(flat_out, x_aug_1, 1)
            if squeeze:
                out = tf.squeeze(out, [len(x_aug_1.get_shape().as_list()) - 1])

            return out

    def encode(self, inputs):
        pass


    def __call__(self):
        # init placeholder
        self.p_c = placeholder(name='p', shape=(None, self.max_char_len), dtype=tf.int32)
        self.h_c = tf.placeholder(name='h', shape=(None, self.max_char_len), dtype=tf.int32)
        self.p_w = tf.placeholder(name='p_word', shape=(None, self.max_word_len), dtype=tf.int32)
        self.h_w = tf.placeholder(name='h_word', shape=(None, self.max_word_len), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.char_embedding_table = tf.get_variable(name='char_embed', shape=(self.char_vocab_len, self.char_embedding_dim), 
                                                    dtype=tf.float32)
        self.word_embedding_table = tf.get_variable(name='word_embed', initializer=word_embedding, dtype=tf.float32, 
                                                    trainable=False)


        # embedding layer
        ## char embedding
        ## 原文中使用的是CNN进行char embedding，这里直接使用查表方式;
        p_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.p_c)
        h_char_embedding = tf.nn.embedding_lookup(self.char_embedding_table, self.h_c)

        ## word embedding 
        ## 使用预训练好的word embedding；
        p_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.p_w)
        h_word_embedding = tf.nn.embedding_lookup(self.word_embedding_table, self.h_w)

        ## 原文中还使用了POS、exact emedding, 此处暂时不加;

        # concat: 正常情况下应该在axis=-1维度进行concat, 例如pos信息等, 
        # 但此处word_length 与 char_length不同, 所以在axis=1进行concat，
        # 保持embedding_dim维度相同;
        # 这也是跟原文有所区别的，原文是在axis=-1上进行concat的, concat后
        # 的结果保持是(b, seq_len, *) 的维度;
        p = tf.concat([p_char_embedding, p_word_embedding], axis=1)
        h = tf.concat([h_char_embedding, h_word_embedding], axis=1)

        # highway
        p = self.highway(p)
        h = self.highway(h)

        # encode layer
        ## self attention 的另外一种方式;
        p = self.dropout(p)
        h = self.dropout(h)




