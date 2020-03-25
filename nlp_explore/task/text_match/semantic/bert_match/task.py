#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：task.py
#   创 建 者：YuLianghua
#   创建日期：2020年03月25日
#   描    述：
#
#================================================================

import tensorflow as tf

class TextMatchTask(object):
  def __init__(self, 
               bert_model, 
               labels,
               num_labels,
               is_traing=True,
               keep_prob=0.9
               ):
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use bert_model.get_sequence_output()
    # instead.
    output_layer = bert_model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

      logits = tf.matmul(output_layer, output_weights, transpose_b=True)
      self.logits = tf.nn.bias_add(logits, output_bias)
      self.probabilities = tf.nn.softmax(logits, axis=-1)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      self.per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      self.loss = tf.reduce_mean(per_example_loss)

