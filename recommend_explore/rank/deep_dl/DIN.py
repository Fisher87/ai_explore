#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：DIN.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import tensorflow as tf

class DIN(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs.get("item_count")
        self.cate_count = kwargs.get("cate_count")
        self.embedding_size = kwargs.get("embedding_size")
        self.learning_rate = kwargs.get("learning_rate")

    def __call__(self):
        # init placeholder
        ## x_i 正样本
        self.x_i = tf.placeholder(dtype=tf.int32, shape=[None,], name="item_i")
        self.x_i_cate = tf.placeholder(dtype=tf.int32, shape=[None,], name="item_i_cate")
        ## x_j 负样本
        self.x_j = tf.placeholder(dtype=tf.int32, shape=[None,], name="item_j")
        self.x_j_cate = tf.placeholder(dtype=tf.int32, shape=[None,], name="item_j_cate")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,], name="label")
        self.hist = tf.placeholder(tf.int32, shape=[None, None], name="history")
        self.hist_cate = tf.placeholder(tf.int32, shape=[None, None], name="history_cate")
        ## 
        self.seq_len= tf.placeholder(tf.int32, shape=[None,], name="seq_length")

        # init weight
        self.init_weights()

        item_i_embedding = tf.concat(
            [tf.nn.embedding_lookup(self.item_embedding_table, self.x_i),
             tf.nn.embedding_lookup(self.cate_embedding_table, self.x_i_cate)]
            , axis=-1)     # [B, H]  H = 2*self.embedding_siz
        item_j_embedding = tf.concat(
            [tf.nn.embedding_lookup(self.item_embedding_table, self.x_j),
             tf.nn.embedding_lookup(self.cate_embedding_table, self.x_j_cate)]
            , axis=-1)     # [B, H]

        history_embedding = tf.concat(
            [tf.nn.embedding_lookup(self.item_embedding_table, self.hist), 
             tf.nn.embedding_lookup(self.cate_embedding_table, self.hist_cate)]
            , axis=-1)     # [B, T, H]

        hist = self.attention(item_i_embedding, history_embedding, self.seq_len)   # [B, 1, H]
        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, self.embedding_size*2])      # [B, H]
        hist = tf.layers.dense(hist, self.embedding_size*2)
        user_embedding = hist

        # fcn
        din_i = tf.concat([user_embedding, item_i_embedding], axis=-1)  # [B, 2H]
        din_i = tf.layers.batch_normalization(inputs=din_i)
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name="din_f1_i")
        d_layer_1_i = self.dice(d_layer_1_i, name="dice_1_i")
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name="din_f2_i")
        d_layer_1_i = self.dice(d_layer_2_i, name="dice_2_i")
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name="din_f3_i")  # [B, 1]

        din_j = tf.concat([user_embedding, item_j_embedding], axis=-1)  # [B, 2H]
        din_j = tf.layers.batch_normalization(inputs=din_j)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name="din_f1_j")
        d_layer_1_j = self.dice(d_layer_1_j, name="dice_1_j")
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name="din_f2_j")
        d_layer_1_j = self.dice(d_layer_2_j, name="dice_2_j")
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name="din_f3_j")   # [B, 1]

        ##
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])  # [B]
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])  # [B]

        self.logits = d_layer_3_i

    def init_weights(self):
        self.item_embedding_table = tf.get_variable("item_embeding_table",
                                        [self.item_count, self.embedding_size],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float32)

        self.cate_embedding_table = tf.get_variable("cate_embedding_table",
                                        [self.cate_count, self.embedding_size],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float32)

    def attention(self, queries, keys, keys_length):
        '''
        queries: [B, H]
        keys   : [B, T, H]
        keys_length: [B]
        '''
        queries_hidden_units = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])

        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)   # [B, T, 4H]

        # fcn
        d_layer_1 = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name="att_f1")
        d_layer_2 = tf.layers.dense(d_layer_1, 40, activation=tf.nn.sigmoid, name="att_f2")
        d_layer_3 = tf.layers.dense(d_layer_2, 1, activation=None, name="att_f3")   # [B, T, 1]

        outputs = tf.reshape(d_layer_3, [-1, 1, tf.shape(keys)[1]])   # [B, 1, T]

        # mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)    # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2**32+1)
        outputs = tf.where(keymasks, outputs, paddings)  # [B, 1, T]

        # scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        # activation
        outputs = tf.nn.softmax(outputs) # [B, 1, T]

        # weight sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]

        return outputs

    def dice(self, _x, axis=-1, epsilon=0.0000001, name=''):
        alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],
                             initializer = tf.constant_initializer(0.0),
                             dtype=tf.float32)

        input_shape = _x.get_shape().as_list()
        reduction_axes = list(range(len(input_shape)))

        del reduction_axes[axis] # [0]

        broadcast_shape = [1] * len(input_shape)  #[1, 1]
        broadcast_shape[axis] = input_shape[axis] # [1, hidden_unit_size]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(_x, axis=reduction_axes) # [hidden_unit_size]
        brodcast_mean = tf.reshape(mean, broadcast_shape)  # [1, hidden_unit_size]
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)  # [hidden_unit_size]
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape) #[1, hidden_unit_size]
        x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
        # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
        x_p = tf.sigmoid(x_normed)

        return alphas * (1.0 - x_p) * _x + x_p * _x

