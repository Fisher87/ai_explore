#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：seq2seq_decoder.py
#   创 建 者：YuLianghua
#   创建日期：2020年05月14日
#   描    述：
#
#================================================================

# **************************************************************************************
# 传给CustomHelper的三个函数
def initial_fn():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
    return initial_elements_finished, initial_input

def sample_fn(time, outputs, state):
    # 选择logit最大的下标作为sample
    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
    return prediction_id

def next_inputs_fn(time, outputs, state, sample_ids):
    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
    pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
    # 输入是h_i+o_{i-1}+c_i
    next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
    next_state = state
    return elements_finished, next_inputs, next_state

# 自定义helper
my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

def decode(helper, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        memory = tf.transpose(encoder_outputs, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.hidden_size, memory=memory,
            memory_sequence_length=self.encoder_inputs_actual_length)
        cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=self.hidden_size)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, self.slot_size, reuse=reuse
        )
        # 使用自定义helper的decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state(
                dtype=tf.float32, batch_size=self.batch_size))
        # 获取decode结果
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=True,
            impute_finished=True, maximum_iterations=self.input_steps
        )
        return final_outputs

outputs = decode(my_helper, 'decode')
# **************************************************************************************


def decode(self, encoder_state):
    tokens_go = tf.ones([batch_size], dtype=tf.int32) * w2i_target['_GO']
    decoder_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim]), dtype=tf.float32)
    decoder_cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
    if use_teacher_forcing:
        decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.seq_targets[:, :-1]], 1)
        decoder_input_embedding = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedding, self.seq_target_length)
    else:
        # GreedyEmbeddingHelper(embedding, start_tokens, end_token)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, w2i_target["_EOS"])
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, 
                                              output_layer=tf.layers.Dense(vocab_size))
    decoder_ouputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                             maximum_iterations=tf.reduce_max(self.seq_targets_length))

def attention(self, encoder_outputs, encoder_state):
    decoder_cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
    if use_attention:
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=hidden_dim,
                                                                  memory=encoder_outputs,
                                                                  memory_sequence_length=self.seq_inputs_length)
        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, 
        #                                                         memory=encoder_outputs, 
        #                                                         memory_sequence_length=self.seq_inputs_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
        decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                             output_layer=tf.layers.Dense(vocab_size))
    decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                             maximum_iterations=tf.reduce_max(self.seq_targets_length))

def beam_search(self, beam_width):
    '''
    beam_width:topk
    '''
    tokens_go = tf.ones([batch_size], dtype=tf.int32) * w2i_target['_GO']
    decoder_cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
    if beam_width> 1:
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, 
                                                             multiplier=beam_width)
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, 
                                                       decoder_embedding, 
                                                       tokens_go, 
                                                       w2i_target["_EOS"], 
                                                       decoder_initial_state , 
                                                       beam_width=beam_width, 
                                                       output_layer=tf.layers.Dense(config.target_vocab_size))
    else:
        decoder_initial_state = encoder_state
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                  helper, 
                                                  decoder_initial_state, 
                                                  output_layer=tf.layers.Dense(config.target_vocab_size))

    decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                               maximum_iterations=tf.reduce_max(self.seq_targets_length))

def seq_loss(self):
    decoder_logits = tf.layers.dense(decoder_outputs.rnn_output, config.target_vocab_size)
    sequence_mask = tf.sequence_mask(self.seq_target_length, dtype=tf.float32)
    self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, target=self.seq_targets, weights=sequence_mask)


