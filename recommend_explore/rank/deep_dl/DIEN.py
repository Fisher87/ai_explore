#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：DIEN.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月18日
#   描    述：
#
#================================================================

from tensorflow as tf

from rnn import dynamic_rnn
from .dien_ops import VecAttGRUCell
from .dien_ops import prelu, dice

class DIEN(object):
    def __init__(self, **kwargs):
        self.user_nums = kwargs.get("user_nums")
        self.item_nums = kwargs.get("item_nums")
        self.cat_nums  = kwargs.get("cat_nums")
        self.embeding_size = kwargs.get("embedding_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.use_negsampling = kwargs.get("use_negsampling")
        self.keep_prob = kwargs.get("keep_prob")
        self.learning_rate = kwargs.get("learning_rate")
        self.attention_size= kwargs.get("attention_size")

    def embedding(self):
        with tf.variable_scope("embedding_layer"):
            # user
            self.user_embedding_table = tf.get_variable('user_embedding_table',
                                                       [self.user_nums, self.embeding_size])
            self.user_embeddings = tf.nn.embedding_lookup(self.user_embedding_table, self.users)

            # item
            self.item_embedding_table = tf.get_variable("item_embedding_table",
                                                       [self.item_nums, self.embeding_size])
            self.item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.target_x)
            self.hist_item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.history_x)
            if self.use_negsampling:
                self.neg_item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.neg_x)

            # cat
            self.cat_embedding_table = tf.get_variable("cat_embedding_table", 
                                                      [self.cat_nums, self.embeding_size])
            self.cat_embeddings = tf.nn.embedding_lookup(self.cat_embedding_table, self.target_cat)
            self.hist_cat_embeddings = tf.nn.embedding_lookup(self.cat_embedding_table, self.history_cat)
            if self.use_negsampling:
                self.neg_cat_embeddings = tf.nn.embedding_lookup(self.cat_embedding_table, self.neg_cat)

        self.x_embedding = tf.concat([self.item_embeddings, self.cat_embeddings], axis=1)
        self.hist_x_embedding = tf.concat([self.hist_item_embeddings, self.hist_cat_embeddings], axis=2)

        if self.use_negsampling:
            # 负采样的item选第一个
            self.neg_x_embedding = tf.concat(
                [self.neg_item_embeddings[:, :, 0, :], self.neg_cat_embeddings[:, :, 0, :]], axis=-1
            ) 
            self.neg_x_embedding = tf.reshape(self.neg_x_embedding, 
                                             [-1, self.neg_item_embeddings.shape[1], self.embeding_size*2])
            self.neg_hist_x_embedding = tf.concat([self.neg_item_embeddings, self.neg_cat_embeddings], -1)

    def auxiliary_loss(self, h_states, click_seq, no_click_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input = tf.concat([h_states, click_seq], -1)
        noclick_input = tf.concat([h_states, no_click_seq], -1)
        click_prop_ = self.auxiliary_net(click_input, stag=stag)[:,:,0]
        noclick_prop_ = self.auxiliary_net(noclick_input, stag=stag)[:,:,0]
        click_loss_ = -tf.reshape(tf.log(click_prop_), [-1, click_seq.shape[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_), [-1, noclick_input.shape[1]]) * mask
        loss_ tf.reduce_mean(click_loss_ + noclick_loss_)

        return loss_

    def auxiliary_net(self, input, stag="auxiliary_net"):
        bn1 = tf.layers.batch_normalization(inputs=input, name="bn1"+stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=tf.nn.sigmoid, name="f1"+stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.dropout(dnn1, keep_prob=self.keep_prob)
        dnn2 = tf.layers.dense(dnn1, 50, activation=tf.nn.sigmoid, name="f2"+stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.dropout(dnn2, keep_prob=self.keep_prob)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name="f3"+stag, reuse=tf.AUTO_REUSE)
        y_hat= tf.nn.softmax(dnn3) + 0.00000001

        return y_hat

    def fcn_attention(self, query, facts, attention_size, mask, 
                      stag="null", mode="SUM", softmax_stag=1, time_major=False, 
                      return_alphas=False, for_cnn=False):
        if isinstance(facts, tuple):
            # bi-rnn
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 2)

        if time_major:
            # (T, B, D) -> (B, T, D)
            facts = tf.transpose(facts, [1, 0, 2])

        mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[-1]  # hidden size for rnn layer
        query = tf.layers.dense(query, facts_size, activation=tf.nn.relu, name="f1"+stag)
        query = tf.nn.dropout(query, keep_prob=self.keep_prob)

        query = tf.expand_dims(query, 1)
        queries = tf.tile(query, [1, tf.shape(facts)[1], 1])  # Batch * Time * Hidden size
        din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1) # Batch * Time * (4 * Hidden size)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag) # Batch * Time * 1

        d_layer_3_all = tf.reshape(d_layer_3_all,[-1,1,tf.shape(facts)[1]])   # Batch * 1 * time
        scores = d_layer_3_all

        key_masks = tf.expand_dims(mask, 1)  # Batch * 1 * Time
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)

        if not for_cnn:
            scores = tf.where(key_masks, scores, paddings)

        if softmax_stag:
            scores = tf.nn.softmax(scores)

        if mode=="SUM":
            output = tf.matmul(scores, facts)
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])  # Batch * Time
            output = facts * tf.expand_dims(scores, -1)  # Batch * Time * Hidden size
            output = tf.reshape(output,tf.shape(facts))

        if return_alphas:
            return output, scores
        else:
            return output


    def __call__(self):
        # init placeholder
        ## 用户历史 item id 
        self.history_x = tf.placeholder(dtype=tf.int32, [None, None], name="history_items") ## 用户历史 item 应的cate id list self.history_cat = tf.placeholder(dtype=tf.int32, [None, None], name="history_cats")
        ## 用户id
        self.users = tf.placeholder(dtype=tf.int32, [None,], name="user_ids")
        ## target的item id
        self.target_x = tf.placeholder(dtype=tf.int32, [None,], name="target_x")
        ## target item对应的cate id
        self.target_cat=tf.placeholder(dtype=tf.int32, [None,], name="target_cat")
        ## 历史行为的mask
        self.mask = tf.placeholder(dtype=tf.float32, [None, None], name="mask")
        ## 历史行为的长度
        self.seq_len = tf.placeholder(dtype=tf.int32, [None], name="seq_len")
        ## 目标值
        self.target = tf.placeholder(dtype=tf.float32, [None, None], name="target")
        
        if self.use_negsampling:
            ## 负采样数据 batch * seq_len * 采样数量
            self.neg_x = tf.placeholder(dtype=tf.int32, [None, None, None], name="neg_items")
            self.neg_cat= tf.placeholder(dtype=tf.int32, [None, None, None], name="neg_cats")

        self.embedding()

        # build graph
        with tf.name_scope("rnn_layer_1"):
            rnn_outputs, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(self.hidden_size), 
                                            inputs=self.hist_x_embedding, 
                                            sequence_length=self.seq_len, 
                                            dtype=tf.float32,
                                            name="gru1")
        # 辅助loss
        self.aux_loss = self.auxiliary_loss(rnn_outputs[:,:-1,:], 
                                       self.hist_x_embedding[:,1:,:],
                                       self.neg_x_embedding[:, 1:,:],
                                       self.mask[:, 1:], 
                                       stag="gru")

        with tf.name_scope("attention_layer_1"):
            att_outputs, alphas = self.fcn_attention(self.x_embedding, rnn_outputs, self.attention_size,
                                               self.mask, softmax_stag=1, stag="1_1", mode="LIST",
                                               return_alphas=True)
        with tf.name_scope("rnn_2"):
            augru_ouputs, final_state = dynamic_rnn(VecAttGRUCell(self.hidden_size),
                                                   inputs=rnn_outputs,
                                                   att_score)
        inp = tf.concat([self.user_embeddings, 
                         self.x_embedding, 
                         self.hist_x_embedding, 
                         self.x_embedding*self.hist_x_embedding, 
                         final_state], axis=1)
        self.fcn_net(inp, use_dice=True)

    def fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp,name='bn1')
        dnn1 = tf.layers.dense(bn1,200,activation=None,name='f1')

        if use_dice:
            dnn1 = dice(dnn1,name='dice_1')
        else:
            dnn1 = prelu(dnn1,'prelu1')

        dnn2 = tf.layers.dense(dnn1,80,activation=None,name='f2')
        if use_dice:
            dnn2 = dice(dnn2,name='dice_2')
        else:
            dnn2 = prelu(dnn2,name='prelu2')

        dnn3 = tf.layers.dense(dnn2,2,activation=None,name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            ctr_loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat),self.target),tf.float32))



