#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：_rnn.py
#   创 建 者：YuLianghua
#   创建日期：2020年08月07日
#   描    述：
#
#================================================================


class _Linear(object):
    def __call__(self, args):
        if not self._is_sequence:
            args = [args]
        if len(args) == 1:
            res = math_ops.matmul(args[0], self._weights)
        else:
            res = math_ops.matmul(array_ops.concat(arg, 1), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res

class RNNCell(object):
    pass

class BasicRNNCell(RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(BasicRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        if self._linear is None:
            self._linear = _Linear([inputs, state], self._num_units, True)
        output = self._activation(self._linear([inputs, state]))
        return output, output


class BasicLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0, 
                 state_is_tuple=True, activation=None, reuse=None):
        super(BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logger.warning()
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None

    @property
    def state_size(self):
        # 对于LSTM其输出的state需要将C_t 和 h_t 合并，所以它需要包含两部分内容;
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        # gate->sigmoid
        sigmoid = math_ops.sigmoid
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)
        # 遗忘门: f  输入门: i  输出门: o
        i, j, f, o = array_ops.split(
            value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        # 加了一个 _forget_bias 变量，即设置了初始化偏置，以免初始输出为 0 的问题
        new_c = ( c*sigmoid(f + self._forget_bias) + sigmoid(i)*self._activation(j) )
        new_h = sigmoid(o) * self._activation(new_c)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)

        return new_h, new_state


class GRUCell(RNNCell):
    def __init__(self, num_units, activation=None,
                 reuse=None, kerner_initializer=None, bias_initializer=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kerner_initializer = kerner_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
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
                    kerner_initializer=self._kerner_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, state], 
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kerner_initializer=self._kerner_initializer)
        h_hate = self._activation(self._gate_linear([inputs, r_state]))
        new_state = (1-u) * state + u*h_hate

