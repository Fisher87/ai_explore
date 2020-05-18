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

import tensorflow as tf

class Seq2SeqWithAtt(object):
    def __init__(self, **kwargs):
        self.max_len = kwargs.max_len
        self.vocab_size = kwargs.vocab_size
        self.word2inx = kwargs.word2inx
        self.embedding_dim= kwargs.embedding_dim
        self.state_size = kwargs.state_size
        self.num_layers = kwargs.num_layers
        self.attention = kwargs.attention
        self.use_teacher_forcing = kwargs.use_teacher_forcing
        self.learning_rate = kwargs.learning_rate
        self.beam_width = kwargs.beam_width

    def multi_lstm_layer_encoder(inputs, lengths, state_size, num_layers,
                                 keep_prob, scope='multi_lstm_layer_encode',
                                 reuse=False):
        """
        Multi LSTM layer encoder
        """
	def single_lstm_cell(state_size, keep_prob, reuse=False):
	    single_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
		tf.contrib.rnn.core_rnn_cell.LSTMCell(
		    state_size,
		    reuse=reuse
		),
		output_keep_prob=keep_prob
	      )
	    return single_cell
        cell = tf.contrib.rnn.MultiRNNCell(
                   [single_lstm_cell(state_size, keep_prob) \
                     for _ in range(num_layers)]
               )

        outputs, output_state = tf.nn.dynamic_rnn(
                inputs=inputs,
                cell  =cell,
                sequence_length=lengths,
                dtype=tf.float32
            )

        return outputs, output_state

    def build_graph(self):
        self.q = tf.placeholder(tf,int32, shape=[None, self.max_len], name='query') 
        self.q_len = tf.placeholder(tf.int32, shape=[None], name="query_len")
        self.d = tf.placeholder(tf.int32, shape=[None, self.max_len], name="doc")
        self.d_len = tf.placeholder(tf.int32, shape=[None], name="doc_len")
        self.keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_prob")

        with tf.variable_scope("embedding_layer"):
            self.embedding_table= tf.get_variable("embedding_table", 
                                             shape=[self.vocab_size, self.embedding_dim],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)

        self.q_embedding = tf.nn.embedding_lookup(self.embedding_table, self.q)

        # encode
        with tf.variable_scope("encoder"):
            encoder_outputs, encoder_state = self.multi_lstm_layer_encoder(
                                                             self.q_embedding,
                                                             self.q_len,
                                                             self.state_size,
                                                             self.num_layers,
                                                             self.keep_prob
                                                         )

        # decode
        with tf.variable_scope("decoder"):
            tokens_go = tf.ones([self.d.shape[0]], dtype=tf.int32) * self.word2inx['GO']

            # decoder cell
            decoder_cell = tf.nn.rnn_cell.GRUCell(self.state_size) 
            if self.attention:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.state_size,
                                                                          memory=encoder_outputs,
                                                                          memory_sequence_length=self.d_len)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

            # helper
            if self.use_teacher_forcing:
                decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.d], axis=1)
                decoder_input_embedding = tf.nn.embedding_lookup(self.embedding_table, decoder_inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedding, self.d_len)
            else:
                # GreedyEmbeddingHelper(embedding, start_tokens, end_token)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_table, tokens_go, self.word2inx['EOS'])

            if self.attention:
                decoder_initial_state = decoder_cell.zero_state(batch_size=self.d.shape[0], dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, 
                                                      output_layer=tf.layers.dense(self.vocab_size))
            decoder_ouputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                                 maximum_iterations=tf.reduce_max(self.d_len))

            # beam search decoder
            beam_search_decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, 
                                                                              multiplier=self.beam_width)

            self.beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, 
                                                                   decoder_embedding, 
                                                                   tokens_go, 
                                                                   w2i_target["_EOS"], 
                                                                   beam_search_decoder_initial_state , 
                                                                   beam_width=self.beam_width, 
                                                                   output_layer=tf.layers.Dense(config.target_vocab_size))
            self.beam_search_decoder_outputs, \
            self.beam_search_decoder_state,   \
            self.beam_search_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.beam_search_decoder, 
                                                                                        maximum_iterations=     \
                                                                                        tf.reduce_max(self.seq_targets_length))
            self.decoder_predict_decode = self.beam_search_decoder_outputs.predicted_ids

        with tf.variable_scope('loss'):
            outputs = decoder_outputs.rnn_output
            self.decoder_logits = tf.layers.dense(outputs, self.vocab_size)
            sequence_mask = tf.sequence_mask(self.d_len, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits, 
                                                         target=self.d, 
                                                         weights=sequence_mask)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
