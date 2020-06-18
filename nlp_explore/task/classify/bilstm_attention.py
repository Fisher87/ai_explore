#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：bilstm_attention.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月25日
#   描    述：
#
#================================================================

import tensorflow as tf

class BiLSTM_Attention(object):
    def __init__(self, seq_len, 
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 attention_size,
                 label_size,
                 learning_rate,
                 random_embedding=True,
                 word_embedding=None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size= hidden_size
        self.attention_size = attention_size
        self.label_size = label_size
        self.learning_rate = learning_rate
        self.random_embedding = random_embedding
        self.word_embedding = word_embedding

        self._build_graph()

    def _build_graph(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.label_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 

        with tf.name_scope("embedding"):
            if not self.random_embedding:
                self.embedding_table = tf.Variable(tf.cast(self.word_embedding, \
                                                           dtype=tf.float32, name="word2vec"), 
                                                  name="embedding_table")
            else:
                self.embedding_table = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_dim], \
                                                                     -1.0, 1.0), 
                                                   name="embedding_table")
            input_embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_x)

        hiddens, _  = self.bilstm(input_embedding, self.hidden_size, self.seq_len)
        x_embedding = tf.concat(hiddens, axis=-1)   # (batch_size, max_time, hidden_size*2)
        attention_output, alphas= self.attention(x_embedding, self.attention_size)
        attention_output = tf.nn.dropout(attention_output, self.dropout_keep_prob)
        
        # 全连接层
        with tf.variable_scope("full_net"):
            attention_output_shape = attention_output.shape
            W = tf.Variable(tf.random_normal(shape=[attention_output_shape[-1].value, self.label_size]\
                                             , stddev=0.1), 
                            name='W')
            b = tf.Variable(tf.random_normal([self.label_size], stddev=0.1), name="b")
            full_output = tf.matmul(attention_output, W) + b
            self.logits = tf.nn.dropout(full_output, self.dropout_keep_prob)
            self.prob = tf.nn.softmax(self.logits)
            self.prediction = tf.argmax(self.prob, axis=1)

        # y = tf.one_hot(self.input_y, self.label_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,\
                                                       labels=self.input_y)
        self.loss = tf.reduce_mean(loss, axis=0)
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.input_y)
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0, name="acc")

    def attention(self, inputs, attention_size, mask=True):
        with tf.variable_scope("attention"):
            # hidden_size = tf.shape(inputs)[-1]
            hidden_size = inputs.shape[2].value
            W = tf.Variable(tf.random_normal(shape=[hidden_size, attention_size], stddev=0.1))
            b = tf.Variable(tf.random_normal(shape=[attention_size], stddev=0.1))
            u = tf.Variable(tf.random_normal(shape=[1, attention_size], stddev=0.1))

            v = tf.tanh(tf.matmul(inputs, W) + b)  # (batch_size, len, attention_size)
            uv = tf.reduce_sum(tf.matmul(v, tf.transpose(u, [1,0])), axis=-1)
            alphas = tf.nn.softmax(uv, axis=-1)

            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

            return output, alphas

    def rnn_cell(self, hidden_size, name, cell_type='lstm'):
        with tf.variable_scope(name):
            if cell_type=="lstm":
                cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
            else:
                cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)
            return cell 

    def bilstm(self, x, hidden_size, seq_len, cell_type="lstm", name="bilstme"):
        with tf.name_scope(name):
            cell_fw = self.rnn_cell(hidden_size, "fw_"+name) 
            cell_bw = self.rnn_cell(hidden_size, "bw_"+name) 
            hidden_states, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                                         cell_bw=cell_bw, 
                                                                         inputs = x,
                                                                         # sequence_length=seq_len,
                                                                         dtype=tf.float32,
                                                                         scope=name)
            return hidden_states, last_states

