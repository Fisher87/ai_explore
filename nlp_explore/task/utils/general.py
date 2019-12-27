#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：layer_utils.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月20日
#   描    述：
#
#================================================================

import tensorflow as tf
from operator import mul

def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes,
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask

def dropout_layer(inputs, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


def cosine_distance_array(v1, v2, cosine_norm=True, eps=1e-6):
    """
    cosine for array/vector.
    @param v1: 2d-tensor, (dim, 1), e.g.([....., a, 1, d])
    @param v2: 2d-tensor, (dim, 1), e.g.([....., b, a, 1])
    @parma cosine_norm: boolean, wether to normalize.
    """
    cosine_numerator = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1), eps))
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1), eps))

    return cosine_numerator / v1_norm / v2_norm

def cosine_distance_matrix(m1, m2, eps=1e-6):
    '''
    cosine for two matrix.
    @param m1: 3d-tensor, (batch, l1, dim).
    @param m2: 3d-tensor, (batch, l2, dim).
    '''
    cosine_numerator = tf.matmul(m1, tf.transpose(l2, perm=[0, 2, 1]))
    norm = tf.norm(m1, axis=-1, keep_dims=True) * tf.transpose(
                               tf.norm(m2, axis=-1, keep_dims=True), perm=[0,2,1])
    cosine = tf.divide(cosine_numerator, norm)

    return cosine
    
def euclidean_distance(v1, v2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance


# tensor 展开与重组
##################################################
def flatten(inputs, keep):
    """
    将输入展开, keep 为保留几个维度不进行展开处理;
    """
    fixed_shape = inputs.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(inputs, out_shape)

    return flat

def reconstruct(inputs, ref, keep):
    """
    与flatten作用相反;
    @param: inputs, 经过flatten展开后的结果;
    @param: ref, 未flatten处理的原始数据;
    """
    ref_shape = ref.get_shape().as_list()
    input_shape = inputs.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    input_start = len(input_shape) - keep
    pre_shape = [ref_shape[i] for i in range(ref_stop)]
    keep_shape= [input_shape[i] for i in range(input_start, len(input_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(inputs, target_shape)

    return out
##################################################

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: (optional) Variable scope to create parameters in.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                           "but saw %d" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(array_ops.concat(1, args), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)

# tensor shape
##################################################
def shape(tensor, dim=None):
    '''
    get tensor shape/dimension as list/int.
    '''
    if not dim:
        return tensor.shape.as_list()
    if dim:
        return tensor.shape.as_list()[dim]

