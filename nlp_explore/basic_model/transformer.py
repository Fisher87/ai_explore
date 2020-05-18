#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：transformer.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月26日
#   描    述：
#
#================================================================

import tensorflow as tf

def position_encoding(inputs, maxlen):
    E = inputs.get_shape().as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope("position_encoding"):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        position_enc = np.array([
            [pos/np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        # masking
        outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def mask(inputs, key_masks=None, type=None):
    '''
    @param: inputs, (B*h, T_q, T_k)
    '''
    padding_num = -2**32 + 1
    if type=="key":
        key_masks = tf.to_float(key_masks)   # (B, T)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0]//tf.shape(key_masks)[0], 1]) # (B*h, T)
        key_masks = tf.expand_dims(key_masks, 1)  # (B*h, 1, T)
        outputs = inputs + key_masks * padding_num
    elif type=="future":
        diag_vals = tf.ones_like(inputs[0, :, :])  #(T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * padding_num
        outpus = tf.where(tf.equal(future_masks, 0), paddings, inputs)

    return outputs

def ln(inputs, scope="layer_normalize"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

def multihead_attention(queries, keys, values, key_masks, 
                        num_heads=8, 
                        dropout_rate=0, 
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''
    @param: queries, (B, T, D)
    '''
    # D scalar
    d = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q_K_V = tf.layers.dense(queries, 3*d, use_bias=True)
        Q,K,V = tf.split(Q_K_V, 3, axis=-1)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  #(B*h, T, d/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        align = tf.matmul(Q_, tf.transponse(K_, [0, 2, 1]))     #(B*h, T, T)
        align /= Q_.get_shape().as_list()[-1] ** 0.5
        align = mask(align, key_masks=key_masks, type="key")
        if causality:
            align = mask(align, key_masks=key_masks, type="future")

        # softmax
        outputs = tf.nn.softmax(align)
        attention = tf.transpose(outputs, [0, 2, 1])
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # residual connection
        outputs += queries
        # layer normalize
        outputs = ln(outputs)

    return outputs

def ff(inputs, num_units, scope='ff'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        # residual connection
        outputs += inputs
        outputs = ln(outputs)

    return outputs


class Transformer(object):
    def __init__(self, config, vocab2id, id2vocab, embeddings):
        self.config = config
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab
        self.embeddings = embeddings

    def encode(self, xs, training=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)
            enc *= enc.get_shape().as_list()[-1] ** 0.5

            enc += position_encoding(enc, self.config.maxlen)
            enc = tf.layers.dropout(enc, self.config.dropout_rate, training=training)

            for i in range(self.config.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # self attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.dropout_rate,
                                              training=training,
                                              causality=False)

                    # ff
                    enc = ff(enc, num_units=[self.config.d_ff, enc.get_shape().as_list()[-1]])
        memory = enc
        return memory, sents1, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs.
        '''
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # target mask
            tgt_masks = tf.math.equal(decoder_inputs, 0)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
            dec *= dec.get_shape().as_list()[-1] ** 0.5 

            dec += position_encoding(dec, maxlen=self.config.maxlen)
            dec = tf.layers.dropout(dec, self.config.dropout_rate, training=training)

            # blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # self attention
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,#用哪个values就用其对应的masks
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope='decoder_self_attention')

                    # vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope='decoder_vanilla_attention')

                    # ff
                    dec = ff(dec, num_units=[self.config.d_ff, dec.get_shape().as_list()[-1]])

            # linear projection
            weights = tf.transpose(self.embeddings)
            logits = tf.einsum('ntd,dk->ntk', dec, weights)
            y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

            return logits, y_hat, y, sents2


