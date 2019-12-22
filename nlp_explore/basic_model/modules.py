#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：modules.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月19日
#   描    述：
#          reference: [1]. https://www.github.com/kyubyong/transformer.
#
#================================================================

import numpy as np
import tensorflow as tf

def position_encoding(x, maxlen, masking=True, scope="position_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''
    E = x.shape[-1]
    N, T = x.shape[0], x.shape[1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        _index = tf.expand_dims(tf.range(T), 0)
        position_ind = tf.tile(_index, axis=[N, 1])

        # first part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2*i/E) for i in range(E)] 
                  for pos in range(maxlen)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # position embedding table
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        
        # mask
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)

        return tf.to_float(outputs)
        
def scaled_dot_product_attention(Q, K, V, key_masks, 
                                causality=False, dropout_rate=0.,
                                training=True, 
                                scope="scaled_dot_product_attention"):
    """
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = K.shape[-1]
        
        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))       # (N, T_q, T_k)
        # scale
        outputs = tf.divide(ouputs, tf.sqrt(d_k))

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(ouputs)       # (N, T_q, T_k)
        attention = tf.transpose(outputs, [0, 2, 1])

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)       # (N, T_q, d_v)

    return outputs

def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2**32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [inputs.shape[0]//key_masks.shape[0], 1])   # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)
        
        # key_masks : set from tf.equal(x, 0), 
        #             if element value `0`, set 1, then to multiply `padding_num`
        outputs = inputs + key_masks * padding_num

    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])     # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [inputs.shape[0], 1, 1])   # (N, T_q, T_k)

        paddings= tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

def multihead_attention(queries, keys, values, key_masks,
                       num_head=8,
                       dropout_rate=0,
                       training=True,
                       causality=False,
                       scope="multihead_attention"):
    """
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: Boolean. If true, units that reference the future are masked.
    """
    d_model = queries.shape[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)     # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)        # (N, K_q, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)      # (N, V_q, d_model)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # (h*N, T_q, d_model)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

        # attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)

        # residual connection
        outputs += queries

        # normalize
        outputs = ln(outputs)

    return outputs

# layer normalization
def ln(x, epsilon=1e-8, scope="ln"):
    """layer normalization."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_shape = x.get_shape()
        params_shape = input_shape[-1:]

        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs-mean) / ((variance+epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs

# learning rate func.
def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''
    Noam scheme learning rate decay.
    @param warmup_steps: scalar. During warmup_steps, learning rate increases 
                                until it reaches init_lr.
    e.g.
    >>> import tensorflow as tf
    >>> global_step = tf.train.get_or_create_global_step()
    >>> lr = noam_scheme(0.0003, global_step, warmup_steps=4000)
    >>> optimizer = tf.train.AdamOptimizer(lr)
    >>> train_op = optimizer.minimize(loss, global_step=global_step)
    '''

    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
