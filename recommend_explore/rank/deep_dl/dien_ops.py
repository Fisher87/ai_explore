#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：dien_ops.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月21日
#   描    述：
#
#================================================================

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import _Linear

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


class VecAttGRUCell(RNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # 一个类实例也可以变成一个可调用对象，只需要实现一个特殊方法__call__()。
    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer
                )

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value, num_or_size_splits=2, axis=1)
        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer
                )

        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h

def dice(_x,axis=-1,epsilon=0.000000001,name=""):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axis = list(range(len(input_shape)))
        del reduction_axis[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    mean = tf.reduce_mean(_x, axis=reduction_axis)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axis)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

