#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：learning_rate.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月20日
#   描    述：
#
#================================================================

import tensorflow as tf

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
