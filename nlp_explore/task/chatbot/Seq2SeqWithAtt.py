#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：model.py
#   创 建 者：YuLianghua
#   创建日期：2020年05月11日
#   描    述：
#
#================================================================
import pdb

import tensorflow as tf

class Seq2SeqWithAtt(object):
    def __init__(self, max_len, vocab_size, word2inx, embedding_dim, state_size, num_layers,
                 use_attention, use_teacher_forcing, learning_rate, beam_width):
        self.max_len    = max_len
        self.vocab_size = vocab_size
        self.word2inx   = word2inx
        self.embedding_dim= embedding_dim
        self.state_size = state_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_teacher_forcing = use_teacher_forcing
        self.learning_rate = learning_rate
        self.beam_width = beam_width
        self.build_graph()

    def multi_lstm_layer(self, state_size, num_layers, keep_prob, 
                         scope='multi_lstm_layer',
                         reuse=False):
        """
        Multi LSTM layer encoder
        """
        def single_lstm_cell(state_size, keep_prob, reuse=False):
            single_cell = tf.contrib.rnn.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(state_size, reuse=reuse),
                            output_keep_prob=keep_prob
                            )
            return single_cell

        cell = tf.contrib.rnn.MultiRNNCell(
                   [single_lstm_cell(state_size, keep_prob) \
                     for _ in range(num_layers)]
               )
        return cell

    def multi_lstm_layer_encoder(self, inputs, lengths, state_size, num_layers,
                                 keep_prob, 
                                 scope='multi_lstm_layer_encode',
                                 reuse=False):
        cell = self.multi_lstm_layer(state_size, num_layers, keep_prob, 
                                     scope=scope, reuse=reuse)
        outputs, output_state = tf.nn.dynamic_rnn(cell, inputs,
                                                  sequence_length=lengths,
                                                  dtype=tf.float32)

        return outputs, output_state

    def build_graph(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.max_len], name='input_x') 
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.max_len], name="input_y")
        # self.x_len = tf.placeholder(tf.int32, shape=[None], name="x_len")
        # self.y_len = tf.placeholder(tf.int32, shape=[None], name="y_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.x_len = tf.count_nonzero(self.input_x, 1, dtype=tf.int32)
        self.y_len = tf.count_nonzero(self.input_y, 1, dtype=tf.int32)

        with tf.variable_scope("embedding_layer"):
            self.embedding_table= tf.get_variable("embedding_table", 
                                             shape=[self.vocab_size, self.embedding_dim],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)

        self.x_embedding = tf.nn.embedding_lookup(self.embedding_table, self.input_x)

        # encode
        with tf.variable_scope("encoder"):
            encoder_outputs, encoder_state = self.multi_lstm_layer_encoder(
                                                             self.x_embedding,
                                                             self.x_len,
                                                             self.state_size,
                                                             self.num_layers,
                                                             self.dropout_keep_prob)

        # decode
        with tf.variable_scope("decoder"):
            tokens_go = tf.ones([tf.shape(self.input_y)[0]], dtype=tf.int32) * self.word2inx['<GO>']
            output_layer = tf.layers.Dense(self.vocab_size, _reuse=tf.AUTO_REUSE)

            # decoder cell
            # decoder_cell = tf.nn.rnn_cell.GRUCell(self.state_size) 
            decoder_cell = self.multi_lstm_layer(self.state_size,
                                                 self.num_layers,
                                                 self.dropout_keep_prob,
                                                 scope='multi_lstm_layer_decoder',
                                                 reuse=False)
            if self.use_attention:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.state_size,
                                                                          memory=encoder_outputs,
                                                                          memory_sequence_length=self.x_len)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
                                                                   attention_mechanism,
                                                                   attention_layer_size=self.state_size, 
                                                                   name='Attention_Wrapper')

            ending = tf.strided_slice(self.input_y, [0, 0], [tf.shape(self.input_y)[0], -1], [1, 1])
            decoder_inputs = tf.concat([tf.fill([tf.shape(self.input_y)[0], 1], self.word2inx['<GO>']), ending], 1)
            # decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.input_y], axis=1)
            decoder_input_embedding = tf.nn.embedding_lookup(self.embedding_table, decoder_inputs)
            # helper
            if self.use_teacher_forcing:
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedding, self.y_len)
            else:
                # GreedyEmbeddingHelper(embedding, start_tokens, end_token)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_table, 
                                                                  tokens_go, 
                                                                  self.word2inx['<EOS>'])

            if self.use_attention:
                decoder_initial_state = decoder_cell.zero_state(batch_size=tf.shape(self.input_y)[0], 
                                                                dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, 
                                                      output_layer=output_layer)
            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                                 maximum_iterations=tf.reduce_max(self.y_len))

            # beam search decoder
            # beam_search_decoder_initial_state = decoder_cell.zero_state(tf.shape(self.input_y)[0] * self.beam_width, tf.float32)

            self.beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                                                   cell = decoder_cell, 
                                                                   embedding = self.embedding_table, 
                                                                   start_tokens = tokens_go, 
                                                                   end_token = self.word2inx["<EOS>"], 
                                                                   # initial_state = beam_search_decoder_initial_state , 
                                                                   initial_state = decoder_initial_state, 
                                                                   beam_width = self.beam_width, 
                                                                   output_layer = output_layer)
            self.beam_search_decoder_outputs, \
            self.beam_search_decoder_state,   \
            self.beam_search_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.beam_search_decoder, 
                                                                                        maximum_iterations=     \
                                                                                        tf.reduce_max(self.x_len))
            self.decoder_predict_decode = self.beam_search_decoder_outputs.predicted_ids

        sequence_mask = tf.sequence_mask(self.y_len, tf.reduce_max(self.y_len), dtype=tf.float32)
        with tf.variable_scope('loss'):
            outputs = decoder_outputs.rnn_output
            # self.decoder_logits = tf.layers.dense(outputs, self.vocab_size)
            self.decoder_logits = tf.identity(outputs)
            self.decoder_predict_train = tf.argmax(self.decoder_logits, axis=-1, name='decoder_pred_train')
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits,
                                                         targets=self.input_y, 
                                                         weights=sequence_mask)

        y_t = tf.argmax(outputs, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.predictions = tf.boolean_mask(y_t, sequence_mask)
        self.mask_label = tf.boolean_mask(self.input_y, sequence_mask)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions, self.mask_label)
            self.correct_index= tf.cast(self.correct_pred, tf.float32)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")
