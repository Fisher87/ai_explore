#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：embedding.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月06日
#   描    述：
#      reference: https://github.com/sjvasquez/quora-duplicate-questions/blob/master/embed.py
#
#================================================================

import tensorflow as tf

from encoder import lstm_encoder
from layer_utils import temporal_convolution_layer

def embedding_from_sparse_encodings(encodings, shape, embedding_matrix=None, scope='gather-embed',
                                    reuse=False):
    """
    Gathers embedding vectors corresponding to values in encodings.  If embedding_matrix is passed,
    then it will be used to initialize the embedding matrix.  Otherwise, the matrix will be
    initialized with random embeddings.
    Args:
        encodings: Tensor of shape [batch_size, sequence length].
        shape: Shape of 2D parameter matrix.  The first dimension should contain
            the vocabulary size and the second dimension should be the size
            of the embedding dimension.
        embedding_matrix: numpy array of the embedding matrix.
    Returns:
        Sequence of embedding vectors.  Tensor of shape [batch_size, sequence length, shape[1]].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=embedding_matrix or tf.contrib.layers.variance_scaling_initializer(),
            shape=shape
        )
        embeddings = tf.nn.embedding_lookup(W, encodings)
        return embeddings

def dense_word_embedding_from_chars(chars, embed_dim, bias=True, scope='dense-word-embed', reuse=False):
    """
    Word embeddings via dense transformation + maxpooling of character sequences.
    Args:
        chars: Tensor of shape [batch_size, word sequence length, char sequence length, alphabet size].
        embed_dim: Dimension of word embeddings.  Integer.
    Returns:
        Sequence of embedding vectors.  Tensor of shape [batch_size, word sequence length, embed_dim].
    """
    with tf.variable_scope(scope, reuse=reuse):
        chars = tf.cast(chars, tf.float32)
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(chars, -1), embed_dim]
        )
        z = tf.einsum('ijkl,lm->ijkm', chars, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[embed_dim]
            )
            z = z + b
        dense_word_embedding = tf.reduce_max(z, 2)
        return dense_word_embedding


def convolutional_word_embedding_from_chars(chars, embed_dim, convolution_width, bias=True,
                                            scope='conv-word-embed', reuse=False):
    """
    Word embeddings via convolution + maxpooling of character sequences.
    Args:
        chars: Tensor of shape [batch_size, word sequence length, char sequence length, alphabet size].
        embed_dim: Dimension of word embeddings  Integer.
        convolution_width:  Number of characters used in the convolution.  Integer.
    Returns:
        Sequence of embedding vectors.  Tensor of shape [batch_size, word sequence length, embed_dim].
    """
    chars = tf.cast(chars, tf.float32)

    # this is super inefficient
    chars = tf.unstack(chars, axis=0)

    conv_word_embeddings = []
    for i, char in enumerate(chars):
        temp_reuse = i != 0 or reuse
        conv = temporal_convolution_layer(
            char, embed_dim, convolution_width, scope=scope, reuse=temp_reuse)
        embedding = tf.reduce_max(conv, axis=1)
        conv_word_embeddings.append(embedding)
    conv_word_embeddings = tf.stack(conv_word_embeddings, axis=0)

    return conv_word_embeddings

def lstm_word_embedding_from_chars(chars, lengths, embed_dim, scope='lstm-word-embed', reuse=False):
    """
    Word embeddings via LSTM encoding of character sequences.
    Args:
        chars: Tensor of shape [batch_size, word sequence length, char sequence length, num characters].
        lengths: Tensor of shape [batch_size, word_sequence length].
        embed_dim: Dimension of word embeddings.  Integer.
    Returns:
        Sequence of embedding vectors.  Tensor of shape [batch_size, word sequence length, embed_dim].
    """
    chars = tf.cast(chars, tf.float32)

    # this is super inefficient
    chars = tf.unstack(chars, axis=0)
    lengths = tf.unstack(lengths, axis=0)

    lstm_word_embeddings = []
    for i, (char, length) in enumerate(zip(chars, lengths)):
        temp_reuse = i != 0 or reuse
        embedding = lstm_encoder(char, length, embed_dim, 1.0, scope=scope, reuse=temp_reuse)
        lstm_word_embeddings.append(embedding)
    lstm_word_embeddings = tf.stack(lstm_word_embeddings, axis=0)

    return lstm_word_embeddings

